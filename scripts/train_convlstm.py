"""
Training Script for Spatial-Temporal Runoff Prediction with ConvLSTM

Features:
- Proper basin shuffling in batches
- Normalization using training statistics
- Learning rate scheduling
- Checkpointing
- Logging with tensorboard
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time

from spatial_dataset import SpatialHydroDataset, get_basin_list
from convlstm_model import SpatialRunoffModel


class Normalizer:
    """
    Handles normalization of input features.
    Computes statistics from training data and applies to train/test.
    """
    
    def __init__(self):
        self.means = None
        self.stds = None
    
    def fit(self, dataset, num_samples=100):
        """
        Compute mean and std from training dataset.
        
        Args:
            dataset: Training dataset
            num_samples: Number of basins to sample for statistics
        """
        print("Computing normalization statistics...")
        
        all_values = []
        num_samples = min(num_samples, len(dataset))
        
        # Sample random basins
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for idx in tqdm(indices, desc="Sampling basins"):
            x, _, _, _ = dataset[idx]  # x shape: (seq_len, channels, H, W)
            all_values.append(x.numpy())
        
        # Stack all samples: (num_samples * seq_len, channels, H, W)
        all_values = np.concatenate(all_values, axis=0)
        
        # Compute per-channel statistics
        # Mean and std over (samples, height, width) dimensions
        self.means = np.nanmean(all_values, axis=(0, 2, 3))  # Use nanmean to handle NaN
        self.stds = np.nanstd(all_values, axis=(0, 2, 3))    # Use nanstd to handle NaN
        
        # Replace any remaining NaN or zero std with safe values
        self.means = np.nan_to_num(self.means, nan=0.0)
        self.stds = np.nan_to_num(self.stds, nan=1.0)
        
        # Ensure no zero std (would cause division by zero)
        self.stds = np.maximum(self.stds, 1e-8)
        
        print(f"Normalization stats computed from {num_samples} basins")
        print(f"  Mean range: [{self.means.min():.4f}, {self.means.max():.4f}]")
        print(f"  Std range: [{self.stds.min():.4f}, {self.stds.max():.4f}]")
        
        # Check for any problematic channels
        problematic = np.where((self.stds < 1e-6) | np.isnan(self.means) | np.isnan(self.stds))[0]
        if len(problematic) > 0:
            print(f"  WARNING: Channels with low/zero variance or NaN: {problematic.tolist()}")
            print(f"  These channels may be constant or contain NaN values")
    
    def normalize(self, x):
        """
        Normalize input tensor.
        
        Args:
            x: (batch, seq_len, channels, H, W) or (seq_len, channels, H, W)
        
        Returns:
            Normalized tensor (same shape as input)
        """
        if self.means is None or self.stds is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Reshape stats for broadcasting
        if x.dim() == 5:  # (batch, seq_len, channels, H, W)
            means = torch.from_numpy(self.means).view(1, 1, -1, 1, 1).to(x.device)
            stds = torch.from_numpy(self.stds).view(1, 1, -1, 1, 1).to(x.device)
        elif x.dim() == 4:  # (seq_len, channels, H, W)
            means = torch.from_numpy(self.means).view(1, -1, 1, 1).to(x.device)
            stds = torch.from_numpy(self.stds).view(1, -1, 1, 1).to(x.device)
        else:
            raise ValueError(f"Unexpected input dimensions: {x.dim()}")
        
        return (x - means) / stds
    
    def save(self, path):
        """Save normalization statistics."""
        np.savez(path, means=self.means, stds=self.stds)
        print(f"Saved normalization stats to {path}")
    
    def load(self, path):
        """Load normalization statistics."""
        data = np.load(path)
        self.means = data['means']
        self.stds = data['stds']
        print(f"Loaded normalization stats from {path}")


def train_epoch(model, dataloader, criterion, optimizer, normalizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (x, y, mask, basin_ids) in enumerate(pbar):
        # Move to device
        x = x.to(device)  # (batch, seq_len, channels, H, W)
        y = y.to(device)  # (batch, seq_len)
        mask = mask.to(device)  # (batch, H, W)
        
        # Normalize inputs
        x = normalizer.normalize(x)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(x, mask)  # (batch, 1)
        
        # Take last timestep of target for prediction
        # (You can also predict entire sequence - modify as needed)
        target = y[:, -1].unsqueeze(1)  # (batch, 1)
        
        # Compute loss
        loss = criterion(predictions, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, normalizer, device):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y, mask, basin_ids in tqdm(dataloader, desc="Validating"):
            # Move to device
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # Normalize inputs
            x = normalizer.normalize(x)
            
            # Forward pass
            predictions = model(x, mask)
            target = y[:, -1].unsqueeze(1)
            
            # Compute loss
            loss = criterion(predictions, target)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    # Compute additional metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # NSE (Nash-Sutcliffe Efficiency)
    numerator = np.sum((all_targets - all_predictions) ** 2)
    denominator = np.sum((all_targets - np.mean(all_targets)) ** 2)
    nse = 1 - (numerator / denominator)
    
    return avg_loss, nse


def main(args):
    """Main training function."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load basin lists
    print("Loading basin lists...")
    train_basins = get_basin_list(args.train_basin_file)
    test_basins = get_basin_list(args.test_basin_file)
    print(f"Train basins: {len(train_basins)}")
    print(f"Test basins: {len(test_basins)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SpatialHydroDataset(
        basin_list=train_basins,
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        train=True,
        mask_channel=args.mask_channel,
        seed=args.seed,
        time_range=None  # Use all training data (10 years)
    )
    
    # Validation dataset: Use FIRST 5 years of test period (days 0-1824)
    val_dataset = SpatialHydroDataset(
        basin_list=test_basins,
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        train=False,
        mask_channel=args.mask_channel,
        seed=args.seed,
        time_range=(0, 1825)  # First 5 years for validation
    )
    
    # Create data loaders with shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # CRITICAL: Shuffle basins so model doesn't learn basin order
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create normalizer and fit on training data
    print("\nSetting up normalization...")
    normalizer = Normalizer()
    normalizer.fit(train_dataset, num_samples=args.norm_samples)
    normalizer.save(output_dir / 'normalizer.npz')
    
    # Create model
    print("\nCreating model...")
    model = SpatialRunoffModel(
        input_channels=32,
        hidden_dims=args.hidden_dims,
        kernel_sizes=args.kernel_sizes,
        mlp_hidden_dims=args.mlp_hidden_dims,
        dropout=args.dropout
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Tensorboard writer
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, 
            normalizer, device, epoch
        )
        
        # Validate
        val_loss, val_nse = validate(
            model, test_loader, criterion, normalizer, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('NSE/val', val_nse, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs} [{epoch_time:.1f}s]")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val NSE:    {val_nse:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        
        # Check if LR was reduced
        if new_lr < old_lr:
            print(f"  ? Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_nse': val_nse,
            'args': vars(args)
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
            print(f"  ? Saved best model (val_loss: {val_loss:.6f})")
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
    parser = argparse.ArgumentParser(description='Train ConvLSTM for runoff prediction')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base data directory')
    parser.add_argument('--train_basin_file', type=str, required=True,
                        help='File containing training basin IDs')
    parser.add_argument('--test_basin_file', type=str, required=True,
                        help='File containing test basin IDs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--seq_length', type=int, default=365,
                        help='Sequence length (days)')
    parser.add_argument('--mask_channel', type=int, default=31,
                        help='Channel index for basin mask (0-indexed)')
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 64],
                        help='Hidden dimensions for ConvLSTM layers')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[5, 3, 3],
                        help='Kernel sizes for ConvLSTM layers')
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[128, 64],
                        help='Hidden dimensions for MLP')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (number of basins per batch)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--norm_samples', type=int, default=100,
                        help='Number of basins to sample for normalization stats')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    main(args)