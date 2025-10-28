"""
Custom Dataset for Spatial-Temporal Hydrological Data (ConvLSTM)

This dataset handles:
- Loading .npy files (61, 61, 32, 3650) for inputs
- Loading .csv files for runoff targets
- Random temporal window sampling
- Basin-level batching with shuffling
- Lazy loading to handle large files
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random


class SpatialHydroDataset(Dataset):
    """
    Dataset for spatial-temporal hydrological modeling.
    
    Each basin has:
    - Input: (61, 61, 32, 3650) numpy array
    - Target: (3650,) runoff timeseries
    - Mask: (61, 61) binary mask (extracted from inputs)
    
    Returns random temporal windows of specified sequence length.
    """
    
    def __init__(
        self,
        basin_list,
        data_dir,
        seq_length=365,
        train=True,
        mask_channel=31,  # Which channel is the mask (0-indexed)
        seed=None,
        time_range=None  # NEW: (start_day, end_day) or None for all days
    ):
        """
        Args:
            basin_list: List of basin IDs (HUC10 codes)
            data_dir: Base path (e.g., '/icebox/data/shares/mh2/mosavat/Distributed')
            seq_length: Number of timesteps per sequence (default: 365)
            train: If True, load training data; else test data
            mask_channel: Which channel index contains the basin mask
            seed: Random seed for reproducibility
            time_range: Tuple (start_day, end_day) to extract temporal subset.
                       E.g., (0, 1825) for first 5 years, (1825, 3650) for last 5 years.
                       None means use all available days.
        """
        self.basin_list = basin_list
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.train = train
        self.mask_channel = mask_channel
        self.time_range = time_range  # NEW
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Set up paths
        if train:
            self.input_dir = self.data_dir / 'train_final_inputs'
            self.target_dir = self.data_dir / 'train_targets'
        else:
            self.input_dir = self.data_dir / 'test_final_inputs'
            self.target_dir = self.data_dir / 'test_targets'
        
        # Verify directories exist
        assert self.input_dir.exists(), f"Input directory not found: {self.input_dir}"
        assert self.target_dir.exists(), f"Target directory not found: {self.target_dir}"
        
        # Cache for loaded data (optional - remove if memory constrained)
        self.data_cache = {}
        self.use_cache = False  # Set to True if you want to cache loaded basins
        
        print(f"Initialized {'training' if train else 'test'} dataset")
        print(f"  Basins: {len(self.basin_list)}")
        print(f"  Sequence length: {self.seq_length}")
        if self.time_range:
            print(f"  Time range: days {self.time_range[0]}-{self.time_range[1]} ({self.time_range[1]-self.time_range[0]} days)")
        print(f"  Input dir: {self.input_dir}")
        print(f"  Target dir: {self.target_dir}")
    
    def __len__(self):
        """Number of basins in dataset."""
        return len(self.basin_list)
    
    def _interpolate_static_channel(self, channel_data):
        """
        Interpolate NaN values in a STATIC channel (constant across time).
        
        For MODIS variables (channels 18-22) that don't change over time,
        we only need to interpolate the 2D spatial slice once, then broadcast
        to all timesteps.
        
        Args:
            channel_data: (61, 61, 3650) array where all timesteps are identical
        
        Returns:
            Interpolated array with NaN filled
        """
        # Take first timestep (all timesteps are identical for static variables)
        slice_2d = channel_data[:, :, 0].copy()  # Shape: (61, 61)
        
        # Only interpolate if there are NaN values
        if np.isnan(slice_2d).any():
            # Get coordinates
            y_coords, x_coords = np.meshgrid(np.arange(61), np.arange(61), indexing='ij')
            
            valid_mask = ~np.isnan(slice_2d)
            
            if valid_mask.any():
                # Valid pixel coordinates and values
                valid_points = np.column_stack([
                    y_coords[valid_mask], 
                    x_coords[valid_mask]
                ])
                valid_values = slice_2d[valid_mask]
                
                # NaN pixel coordinates  
                nan_mask = np.isnan(slice_2d)
                nan_points = np.column_stack([
                    y_coords[nan_mask], 
                    x_coords[nan_mask]
                ])
                
                # Interpolate using nearest neighbor
                from scipy.interpolate import NearestNDInterpolator
                interpolator = NearestNDInterpolator(valid_points, valid_values)
                slice_2d[nan_mask] = interpolator(nan_points)
            else:
                # If all NaN, fill with 0
                slice_2d[:] = 0.0
        
        # Broadcast the interpolated 2D slice to all timesteps
        channel_data[:, :, :] = slice_2d[:, :, np.newaxis]
        
        return channel_data
    
    def _load_basin_data(self, basin_id):
        """
        Load input and target data for a single basin.
        
        Returns:
            inputs: (61, 61, 32, 3650) numpy array
            targets: (3650,) numpy array
            mask: (61, 61) binary mask
        """
        # Check cache first
        if self.use_cache and basin_id in self.data_cache:
            return self.data_cache[basin_id]
        
        # Load input .npy file
        input_file = self.input_dir / f"{basin_id}.npy"
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        inputs = np.load(input_file)  # Shape: (61, 61, 32, 3650)
        
        # Interpolate MODIS channels (18-22) - these are STATIC variables
        # They don't change over time, so we only interpolate the 2D slice once
        modis_channels = [18, 19, 20, 21, 22]
        for ch in modis_channels:
            inputs[:, :, ch, :] = self._interpolate_static_channel(inputs[:, :, ch, :])
        
        # Extract mask from specified channel
        mask = inputs[:, :, self.mask_channel, 0]  # Shape: (61, 61)
        # Ensure binary mask
        mask = (mask > 0).astype(np.float32)
        
        # Load target .csv file
        target_file = self.target_dir / f"{basin_id}.csv"
        if not target_file.exists():
            raise FileNotFoundError(f"Target file not found: {target_file}")
        
        target_df = pd.read_csv(target_file)
        targets = target_df['runoff'].values  # Shape: (3650,)
        
        # Verify shapes
        assert inputs.shape == (61, 61, 32, 3650), f"Unexpected input shape: {inputs.shape}"
        assert targets.shape[0] == 3650, f"Unexpected target length: {targets.shape[0]}"
        assert mask.shape == (61, 61), f"Unexpected mask shape: {mask.shape}"
        
        # Cache if enabled
        if self.use_cache:
            self.data_cache[basin_id] = (inputs, targets, mask)
        
        return inputs, targets, mask
    
    def __getitem__(self, idx):
        """
        Get a random temporal window from basin at index idx.
        
        Returns:
            x: (seq_length, 32, 61, 61) - input sequence
            y: (seq_length,) - target runoff sequence
            mask: (61, 61) - basin mask
            basin_id: str - basin identifier
        """
        basin_id = self.basin_list[idx]
        
        # Load basin data
        inputs, targets, mask = self._load_basin_data(basin_id)
        
        # inputs shape: (61, 61, 32, 3650)
        # We need: (seq_length, 32, 61, 61)
        
        total_timesteps = inputs.shape[3]
        
        # Sample random start index for temporal window
        if total_timesteps <= self.seq_length:
            # If data shorter than seq_length, use all
            start_idx = 0
            end_idx = total_timesteps
        else:
            # Random window
            start_idx = np.random.randint(0, total_timesteps - self.seq_length + 1)
            end_idx = start_idx + self.seq_length
        
        # Extract temporal window and transpose to (time, channels, height, width)
        x = inputs[:, :, :, start_idx:end_idx]  # (61, 61, 32, seq_length)
        x = np.transpose(x, (3, 2, 0, 1))  # (seq_length, 32, 61, 61)
        
        # Extract corresponding targets
        y = targets[start_idx:end_idx]  # (seq_length,)
        
        # Convert to PyTorch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        mask = torch.from_numpy(mask).float()
        
        return x, y, mask, basin_id


def get_basin_list(basin_file):
    """
    Load basin list from text file.
    
    Args:
        basin_file: Path to file containing basin IDs (one per line)
    
    Returns:
        List of basin ID strings
    """
    with open(basin_file, 'r') as f:
        basins = [line.strip() for line in f if line.strip()]
    return basins


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    data_dir = "/icebox/data/shares/mh2/mosavat/Distributed"
    
    # Create a small test list
    test_basins = ["0101000101", "0101000102"]  # Replace with actual basin IDs
    
    dataset = SpatialHydroDataset(
        basin_list=test_basins,
        data_dir=data_dir,
        seq_length=365,
        train=True,
        mask_channel=31
    )
    
    # Test loading one sample
    x, y, mask, basin_id = dataset[0]
    
    print(f"\nSample from basin {basin_id}:")
    print(f"  Input shape: {x.shape}")  # Should be (365, 32, 61, 61)
    print(f"  Target shape: {y.shape}")  # Should be (365,)
    print(f"  Mask shape: {mask.shape}")  # Should be (61, 61)
    print(f"  Mask valid pixels: {mask.sum().item()}")
    print(f"  Target range: [{y.min():.6f}, {y.max():.6f}]")
    
    # Check for NaN after interpolation
    nan_count = torch.isnan(x).sum().item()
    print(f"  NaN values in input: {nan_count}")
    if nan_count == 0:
        print("  ? All MODIS channels successfully interpolated!")
    else:
        print(f"  ? Warning: {nan_count} NaN values remaining")