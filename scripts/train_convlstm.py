"""
Copyright (c) 2025, Mohammad Mosavat
Email Address: smosavat@crimson.ua.edu
All rights reserved.

This code is released under MIT License

Description:
OPTIMIZED Training Script for Spatial-Temporal Runoff Prediction with ConvLSTM

Features:
- Mixed precision training (FP16) for ~2x speedup
- Larger batch sizes for better GPU utilization
- More DataLoader workers with persistent_workers
- Gradient accumulation for effective large batch training
- GPU profiling support
- Vectorized operations
- Per-basin validation metrics (median NSE)
- Fixed validation windows for consistent epoch-to-epoch comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time
import csv

from spatial_dataset import SpatialHydroDataset, get_basin_list
from convlstm_model import SpatialRunoffModel


class FixedWindowDataset(SpatialHydroDataset):
    """
    Dataset that returns a FIXED window per basin for validation.
    No randomness - always returns the same window for each basin.
    This ensures validation metrics are comparable across epochs.
    """
    
    def __getitem__(self, idx):
        """Get a FIXED temporal window (no randomness)."""
        basin_id = self.basin_list[idx]
        
        # Load basin data
        inputs, targets, mask = self._load_basin_data(basin_id)
        
        total_timesteps = inputs.shape[3]
        
        # FIXED: Always take the FIRST complete window
        # This ensures validation is consistent across epochs
        if total_timesteps <= self.seq_length:
            start_idx = 0
            end_idx = total_timesteps
            
            x = inputs[:, :, :, start_idx:end_idx]
            x = np.transpose(x, (3, 2, 0, 1))
            y = targets[start_idx:end_idx]
            
            if total_timesteps < self.seq_length:
                pad_length = self.seq_length - total_timesteps
                x = np.pad(x, ((0, pad_length), (0, 0), (0, 0), (0, 0)), mode='constant')
                y = np.pad(y, (0, pad_length), mode='constant')
        else:
            # FIXED: Always use first window (no randomness!)
            start_idx = 0
            end_idx = self.seq_length
            
            x = inputs[:, :, :, start_idx:end_idx]
            x = np.transpose(x, (3, 2, 0, 1))
            y = targets[start_idx:end_idx]
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        mask = torch.from_numpy(mask).float()
        
        return x, y, mask, basin_id


class Normalizer:
    """
    Handles normalization of input features AND targets.
    Optimized for mixed precision training.
    """
    
    def __init__(self):
        self.input_means = None
        self.input_stds = None
        self.target_mean = None
        self.target_std = None
    
    def fit(self, dataset, num_samples=100):
        """
        Compute mean and std from training dataset for BOTH inputs and targets.
        
        Args:
            dataset: Training dataset
            num_samples: Number of basins to sample for statistics
        """
        print("Computing normalization statistics...")
        
        all_input_values = []
        all_target_values = []
        num_samples = min(num_samples, len(dataset))
        
        # Sample random basins
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for idx in tqdm(indices, desc="Sampling basins"):
            x, y, _, _ = dataset[idx]  # x: (seq_len, channels, H, W), y: (seq_len,)
            all_input_values.append(x.numpy())
            all_target_values.append(y.numpy())
        
        # === INPUT NORMALIZATION ===
        # Stack all samples: (num_samples * seq_len, channels, H, W)
        all_input_values = np.concatenate(all_input_values, axis=0)
        
        # Compute per-channel statistics
        self.input_means = np.nanmean(all_input_values, axis=(0, 2, 3))
        self.input_stds = np.nanstd(all_input_values, axis=(0, 2, 3))
        
        # Replace any remaining NaN or zero std with safe values
        self.input_means = np.nan_to_num(self.input_means, nan=0.0)
        self.input_stds = np.nan_to_num(self.input_stds, nan=1.0)
        self.input_stds = np.maximum(self.input_stds, 1e-8)
        
        # === TARGET NORMALIZATION ===
        all_target_values = np.concatenate(all_target_values, axis=0)
        self.target_mean = float(np.nanmean(all_target_values))
        self.target_std = float(np.nanstd(all_target_values))
        
        # Ensure no zero std
        if self.target_std < 1e-8:
            self.target_std = 1.0
        
        print(f"Normalization stats computed from {num_samples} basins")
        print(f"  INPUT - Mean range: [{self.input_means.min():.4f}, {self.input_means.max():.4f}]")
        print(f"  INPUT - Std range: [{self.input_stds.min():.4f}, {self.input_stds.max():.4f}]")
        print(f"  TARGET - Mean: {self.target_mean:.4f}, Std: {self.target_std:.4f}")
    
    def normalize_inputs(self, x):
        """
        Normalize input tensor with NaN handling.
        Optimized for GPU operations.
        """
        if self.input_means is None or self.input_stds is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Handle NaNs BEFORE normalization (replace with 0)
        x = torch.nan_to_num(x, nan=0.0)
        
        # Reshape stats for broadcasting
        if x.dim() == 5:  # (batch, seq_len, channels, H, W)
            means = torch.from_numpy(self.input_means).view(1, 1, -1, 1, 1).to(x.device, x.dtype)
            stds = torch.from_numpy(self.input_stds).view(1, 1, -1, 1, 1).to(x.device, x.dtype)
        elif x.dim() == 4:  # (seq_len, channels, H, W)
            means = torch.from_numpy(self.input_means).view(1, -1, 1, 1).to(x.device, x.dtype)
            stds = torch.from_numpy(self.input_stds).view(1, -1, 1, 1).to(x.device, x.dtype)
        else:
            raise ValueError(f"Unexpected input dimensions: {x.dim()}")
        
        return (x - means) / stds
    
    def normalize_targets(self, y):
        """Normalize target tensor."""
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return (y - self.target_mean) / self.target_std
    
    def denormalize_targets(self, y_norm):
        """Denormalize target tensor back to original scale."""
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return y_norm * self.target_std + self.target_mean
    
    def save(self, path):
        """Save normalization statistics."""
        np.savez(
            path, 
            input_means=self.input_means, 
            input_stds=self.input_stds,
            target_mean=self.target_mean,
            target_std=self.target_std
        )
        print(f"Saved normalization stats to {path}")
    
    def load(self, path):
        """Load normalization statistics."""
        data = np.load(path)
        self.input_means = data['input_means']
        self.input_stds = data['input_stds']
        self.target_mean = float(data['target_mean'])
        self.target_std = float(data['target_std'])
        print(f"Loaded normalization stats from {path}")


def train_epoch(model, dataloader, criterion, optimizer, normalizer, 
                device, epoch, scaler, accumulation_steps=1):
    """
    Train for one epoch with mixed precision and gradient accumulation.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, (x, y, mask, basin_ids) in enumerate(pbar):
        # Move to device
        x = x.to(device)  # (batch, seq_len, channels, H, W)
        y = y.to(device)  # (batch, seq_len)
        mask = mask.to(device)  # (batch, H, W)
        
        # Normalize inputs AND targets
        x = normalizer.normalize_inputs(x)
        y_norm = normalizer.normalize_targets(y)
        
        # Mixed precision forward pass
        with autocast():
            predictions = model(x, mask)  # (batch, seq_len, 1)
            
            # Compute loss on ALL timesteps (not just last)
            # This provides much more training signal!
            loss = criterion(predictions.squeeze(-1), y_norm)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.6f}',
            'avg_loss': f'{total_loss / num_batches:.6f}'
        })
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, normalizer, device):
    """
    Validate model with PER-BASIN metrics.
    Returns median NSE across all basins (more robust than mean).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Store predictions per basin
    basin_predictions = {}
    basin_targets = {}
    
    with torch.no_grad():
        for x, y, mask, basin_ids in tqdm(dataloader, desc="Validating"):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            x = normalizer.normalize_inputs(x)
            y_norm = normalizer.normalize_targets(y)
            
            # Mixed precision inference
            with autocast():
                predictions = model(x, mask)
                loss = criterion(predictions.squeeze(-1), y_norm)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store denormalized predictions PER BASIN
            pred_denorm = normalizer.denormalize_targets(predictions.squeeze(-1))
            
            for i, basin_id in enumerate(basin_ids):
                if basin_id not in basin_predictions:
                    basin_predictions[basin_id] = []
                    basin_targets[basin_id] = []
                
                basin_predictions[basin_id].append(pred_denorm[i].cpu())
                basin_targets[basin_id].append(y[i].cpu())
    
    # Compute NSE per basin
    basin_nse_values = []
    
    for basin_id in basin_predictions.keys():
        # Concatenate all windows for this basin
        preds = torch.cat(basin_predictions[basin_id], dim=0).flatten()
        targets = torch.cat(basin_targets[basin_id], dim=0).flatten()
        
        # Compute NSE for this basin
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        
        if ss_tot > 0:
            nse = 1 - (ss_res / ss_tot)
            basin_nse_values.append(nse.item())
    
    # Return median NSE across basins (more robust than mean)
    median_nse = np.median(basin_nse_values) if basin_nse_values else -float('inf')
    mean_nse = np.mean(basin_nse_values) if basin_nse_values else -float('inf')
    
    print(f"  Per-basin NSE - Median: {median_nse:.4f}, Mean: {mean_nse:.4f}, "
          f"Min: {min(basin_nse_values):.4f}, Max: {max(basin_nse_values):.4f}")
    
    return total_loss / num_batches, median_nse


def profile_gpu_usage():
    """Simple GPU usage profiler."""
    if torch.cuda.is_available():
        print("\nGPU Profile:")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def main(args):
    """Main training function with optimizations."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        # Enable TF32 for additional speedup on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load basin lists
    print("Loading basin lists...")
    train_basins = get_basin_list(args.train_basin_file)
    test_basins = get_basin_list(args.test_basin_file)
    print(f"Training basins: {len(train_basins)}")
    print(f"Test basins: {len(test_basins)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SpatialHydroDataset(
        basin_list=train_basins,
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        train=True,
        mask_channel=args.mask_channel,
        time_range=(0, 3285)  # First 9 years for training (2003-2012)
    )
    
    # FIXED VALIDATION DATASET - uses FixedWindowDataset for consistent validation
    val_dataset = FixedWindowDataset(
        basin_list=test_basins,
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        train=False,
        mask_channel=args.mask_channel,
        time_range=(0, 1825)  # First 5 years for validation (2014-2018)
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Compute normalization statistics
    normalizer = Normalizer()
    normalizer.fit(train_dataset, num_samples=args.norm_samples)
    normalizer.save(output_dir / 'normalizer.npz')
    
    # Create model
    print("Creating model...")
    model = SpatialRunoffModel(
        input_channels=32,
        hidden_dims=args.hidden_dims,
        kernel_sizes=args.kernel_sizes,
        mlp_hidden_dims=args.mlp_hidden_dims,
        dropout=args.dropout,
        use_mask=args.use_mask
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    profile_gpu_usage()
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Gradient accumulation steps
    accumulation_steps = args.gradient_accumulation
    effective_batch_size = args.batch_size * accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'runs')
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {accumulation_steps}")
    print(f"Workers: {args.num_workers}")
    print(f"Mixed precision: Enabled")
    print(f"Validation: Fixed windows with per-basin median NSE")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, 
            normalizer, device, epoch, scaler, accumulation_steps
        )
        
        # Validate
        val_loss, val_nse = validate(model, val_loader, criterion, normalizer, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('NSE/val_median', val_nse, epoch)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val NSE (Median): {val_nse:.6f}")
        
        # Profile GPU usage
        if epoch == 1 or epoch % 10 == 0:
            profile_gpu_usage()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_nse': val_nse,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            print(f"? Saved best model (val_loss: {val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized ConvLSTM training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_basin_file', type=str, required=True)
    parser.add_argument('--test_basin_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seq_length', type=int, default=365)
    parser.add_argument('--mask_channel', type=int, default=31)
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 64])
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[5, 3, 3])
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_mask', action='store_true', help='Use mask-weighted pooling')
    
    # Training arguments - OPTIMIZED DEFAULTS
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (increased from 4!)')
    parser.add_argument('--gradient_accumulation', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--norm_samples', type=int, default=100)
    
    # System arguments - OPTIMIZED
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (increased!)')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    main(args)