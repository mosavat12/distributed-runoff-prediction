"""
Evaluation Script for Trained ConvLSTM Model

Evaluates model on test set and generates:
- Per-basin predictions
- Aggregated metrics (NSE, RMSE, etc.)
- Visualizations
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from spatial_dataset import SpatialHydroDataset, get_basin_list
from convlstm_model import SpatialRunoffModel
from train_convlstm import Normalizer


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: (N,) array
        targets: (N,) array
    
    Returns:
        Dictionary of metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) == 0:
        return None
    
    # NSE (Nash-Sutcliffe Efficiency)
    numerator = np.sum((targets - predictions) ** 2)
    denominator = np.sum((targets - np.mean(targets)) ** 2)
    nse = 1 - (numerator / denominator)
    
    # RMSE
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    
    # MAE
    mae = np.mean(np.abs(targets - predictions))
    
    # R-squared
    correlation_matrix = np.corrcoef(targets, predictions)
    r_squared = correlation_matrix[0, 1] ** 2
    
    # Percent bias
    pbias = 100 * np.sum(predictions - targets) / np.sum(targets)
    
    return {
        'nse': nse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'pbias': pbias,
        'count': len(predictions)
    }


def evaluate_model(model, dataloader, normalizer, device, output_dir):
    """
    Evaluate model on test set.
    
    Returns:
        DataFrame with per-basin predictions and metrics
    """
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for x, y, mask, basin_ids in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # Normalize inputs
            x = normalizer.normalize(x)
            
            # Forward pass
            predictions = model(x, mask)  # (batch, 1)
            
            # Get targets (last timestep)
            targets = y[:, -1].unsqueeze(1)  # (batch, 1)
            
            # Store results for each basin in batch
            for i in range(len(basin_ids)):
                results.append({
                    'basin_id': basin_ids[i],
                    'prediction': predictions[i, 0].cpu().item(),
                    'target': targets[i, 0].cpu().item()
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute per-basin metrics
    basin_metrics = []
    for basin_id in results_df['basin_id'].unique():
        basin_data = results_df[results_df['basin_id'] == basin_id]
        metrics = compute_metrics(
            basin_data['prediction'].values,
            basin_data['target'].values
        )
        if metrics:
            metrics['basin_id'] = basin_id
            basin_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(basin_metrics)
    
    return results_df, metrics_df


def plot_results(results_df, metrics_df, output_dir):
    """
    Create evaluation plots.
    """
    output_dir = Path(output_dir)
    
    # 1. Scatter plot: Predicted vs Observed
    plt.figure(figsize=(8, 8))
    plt.scatter(results_df['target'], results_df['prediction'], 
                alpha=0.5, s=10)
    
    # Add 1:1 line
    min_val = min(results_df['target'].min(), results_df['prediction'].min())
    max_val = max(results_df['target'].max(), results_df['prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 line')
    
    plt.xlabel('Observed Runoff', fontsize=12)
    plt.ylabel('Predicted Runoff', fontsize=12)
    plt.title('Predicted vs Observed Runoff', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_pred_vs_obs.png', dpi=300)
    plt.close()
    
    # 2. NSE distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df['nse'], bins=30, edgecolor='black')
    plt.axvline(metrics_df['nse'].median(), color='red', 
                linestyle='--', label=f'Median: {metrics_df["nse"].median():.3f}')
    plt.xlabel('NSE', fontsize=12)
    plt.ylabel('Number of Basins', fontsize=12)
    plt.title('Distribution of NSE across Basins', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'nse_distribution.png', dpi=300)
    plt.close()
    
    # 3. RMSE vs basin characteristics (if available)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(metrics_df)), metrics_df['rmse'])
    plt.xlabel('Basin Index', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE across Basins', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_distribution.png', dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def main(args):
    """Main evaluation function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint['args']
    
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.6f}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    # Load basin list
    print("Loading basin list...")
    test_basins = get_basin_list(args.test_basin_file)
    print(f"Test basins: {len(test_basins)}")
    
    # Create dataset: Use LAST 5 years of test period (days 1825-3649)
    print("Creating dataset...")
    test_dataset = SpatialHydroDataset(
        basin_list=test_basins,
        data_dir=args.data_dir,
        seq_length=train_args['seq_length'],
        train=False,
        mask_channel=train_args['mask_channel'],
        time_range=(1825, 3650)  # Last 5 years for final testing
    )
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load normalizer
    print("Loading normalizer...")
    normalizer = Normalizer()
    normalizer_path = Path(args.checkpoint).parent / 'normalizer.npz'
    normalizer.load(normalizer_path)
    
    # Create model
    print("Creating model...")
    model = SpatialRunoffModel(
        input_channels=32,
        hidden_dims=train_args['hidden_dims'],
        kernel_sizes=train_args['kernel_sizes'],
        mlp_hidden_dims=train_args['mlp_hidden_dims'],
        dropout=train_args['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    print("\nEvaluating model...")
    results_df, metrics_df = evaluate_model(
        model, test_loader, normalizer, device, output_dir
    )
    
    # Save results
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    metrics_df.to_csv(output_dir / 'basin_metrics.csv', index=False)
    
    # Compute aggregate metrics
    print("\n" + "="*50)
    print("AGGREGATE METRICS")
    print("="*50)
    
    overall_metrics = compute_metrics(
        results_df['prediction'].values,
        results_df['target'].values
    )
    
    for metric, value in overall_metrics.items():
        print(f"{metric.upper():12s}: {value:.6f}")
    
    print("\n" + "="*50)
    print("PER-BASIN METRICS (Summary)")
    print("="*50)
    print(metrics_df[['nse', 'rmse', 'mae', 'r_squared']].describe())
    
    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("OVERALL METRICS\n")
        f.write("="*50 + "\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric.upper():12s}: {value:.6f}\n")
        
        f.write("\n\nPER-BASIN METRICS (Summary)\n")
        f.write("="*50 + "\n")
        f.write(metrics_df[['nse', 'rmse', 'mae', 'r_squared']].describe().to_string())
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(results_df, metrics_df, output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained ConvLSTM model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base data directory')
    parser.add_argument('--test_basin_file', type=str, required=True,
                        help='File containing test basin IDs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)