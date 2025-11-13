"""
Copyright (c) 2025, Mohammad Mosavat
Email Address: smosavat@crimson.ua.edu
All rights reserved.

This code is released under MIT License

Description:
ConvLSTM-based distributed hydrological model for runoff prediction.
Evaluation Script for Trained ConvLSTM Model

Evaluates model on test set and generates:
- Per-basin predictions
- Aggregated metrics (NSE, RMSE, etc.)
- Visualizations

FIXED:
- Uses correct normalizer method names (normalize_inputs, denormalize_targets)
- Denormalizes predictions before computing metrics
- Uses ALL timesteps, not just the last one
- Computes per-basin metrics properly
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
    nse = 1 - (numerator / denominator) if denominator > 0 else -float('inf')
    
    # RMSE
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    
    # MAE
    mae = np.mean(np.abs(targets - predictions))
    
    # R-squared
    correlation_matrix = np.corrcoef(targets, predictions)
    r_squared = correlation_matrix[0, 1] ** 2 if correlation_matrix.size > 1 else 0.0
    
    # Percent bias
    pbias = 100 * np.sum(predictions - targets) / np.sum(targets) if np.sum(targets) != 0 else 0.0
    
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
    Evaluate model on test set with proper denormalization.
    
    Returns:
        DataFrame with per-basin predictions and metrics
    """
    model.eval()
    
    # Store all predictions per basin
    basin_predictions = {}
    basin_targets = {}
    
    with torch.no_grad():
        for x, y, mask, basin_ids in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # Normalize inputs only (not targets - we want original scale for evaluation)
            x = normalizer.normalize_inputs(x)
            
            # Forward pass
            predictions = model(x, mask)  # (batch, seq_len, 1)
            
            # Denormalize predictions to original scale
            predictions_denorm = normalizer.denormalize_targets(predictions.squeeze(-1))  # (batch, seq_len)
            
            # Store ALL timesteps for each basin (not just last!)
            for i, basin_id in enumerate(basin_ids):
                if basin_id not in basin_predictions:
                    basin_predictions[basin_id] = []
                    basin_targets[basin_id] = []
                
                # Store all timesteps
                basin_predictions[basin_id].append(predictions_denorm[i].cpu().numpy())  # (seq_len,)
                basin_targets[basin_id].append(y[i].cpu().numpy())  # (seq_len,)
    
    # Compute per-basin metrics
    basin_metrics = []
    all_predictions = []
    all_targets = []
    
    for basin_id in sorted(basin_predictions.keys()):
        # Concatenate all windows for this basin
        basin_preds = np.concatenate(basin_predictions[basin_id])  # (total_timesteps,)
        basin_targs = np.concatenate(basin_targets[basin_id])  # (total_timesteps,)
        
        # Compute metrics for this basin
        metrics = compute_metrics(basin_preds, basin_targs)
        
        if metrics:
            metrics['basin_id'] = basin_id
            basin_metrics.append(metrics)
            
            # Also store for overall metrics
            all_predictions.extend(basin_preds)
            all_targets.extend(basin_targs)
    
    metrics_df = pd.DataFrame(basin_metrics)
    
    # Create a DataFrame with individual predictions for plotting
    results_list = []
    for basin_id in sorted(basin_predictions.keys()):
        basin_preds = np.concatenate(basin_predictions[basin_id])
        basin_targs = np.concatenate(basin_targets[basin_id])
        
        for pred, targ in zip(basin_preds, basin_targs):
            results_list.append({
                'basin_id': basin_id,
                'prediction': pred,
                'target': targ
            })
    
    results_df = pd.DataFrame(results_list)
    
    # Compute overall metrics
    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    
    return results_df, metrics_df, overall_metrics


def plot_results(results_df, metrics_df, output_dir):
    """
    Create evaluation plots.
    """
    output_dir = Path(output_dir)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Scatter plot: Predicted vs Observed
    plt.figure(figsize=(10, 10))
    
    # Sample points if too many for visualization
    if len(results_df) > 10000:
        plot_df = results_df.sample(n=10000, random_state=42)
    else:
        plot_df = results_df
    
    plt.scatter(plot_df['target'], plot_df['prediction'], 
                alpha=0.3, s=5, c='blue', edgecolors='none')
    
    # Add 1:1 line
    min_val = min(results_df['target'].min(), results_df['prediction'].min())
    max_val = max(results_df['target'].max(), results_df['prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 line')
    
    plt.xlabel('Observed Runoff (mm/day)', fontsize=14)
    plt.ylabel('Predicted Runoff (mm/day)', fontsize=14)
    plt.title('Predicted vs Observed Runoff', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_pred_vs_obs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. NSE distribution
    plt.figure(figsize=(12, 6))
    plt.hist(metrics_df['nse'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    median_nse = metrics_df['nse'].median()
    mean_nse = metrics_df['nse'].mean()
    
    plt.axvline(median_nse, color='red', linestyle='--', linewidth=2, 
                label=f'Median: {median_nse:.3f}')
    plt.axvline(mean_nse, color='orange', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_nse:.3f}')
    
    plt.xlabel('NSE', fontsize=14)
    plt.ylabel('Number of Basins', fontsize=14)
    plt.title('Distribution of NSE across Basins', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'nse_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box plot of metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['nse', 'rmse', 'mae', 'r_squared']
    titles = ['Nash-Sutcliffe Efficiency', 'Root Mean Square Error', 
              'Mean Absolute Error', 'R-squared']
    
    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        ax.boxplot(metrics_df[metric].dropna(), vert=True)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add median line
        median_val = metrics_df[metric].median()
        ax.axhline(median_val, color='red', linestyle='--', linewidth=1, 
                   label=f'Median: {median_val:.3f}')
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. RMSE vs NSE scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(metrics_df['nse'], metrics_df['rmse'], alpha=0.5, s=50, 
                c='steelblue', edgecolors='black', linewidth=0.5)
    plt.xlabel('NSE', fontsize=14)
    plt.ylabel('RMSE (mm/day)', fontsize=14)
    plt.title('RMSE vs NSE across Basins', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_vs_nse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Performance categories
    plt.figure(figsize=(10, 6))
    
    nse_categories = pd.cut(metrics_df['nse'], 
                            bins=[-np.inf, 0, 0.5, 0.65, 0.75, np.inf],
                            labels=['Poor (<0)', 'Unsatisfactory (0-0.5)', 
                                   'Satisfactory (0.5-0.65)', 'Good (0.65-0.75)', 
                                   'Very Good (>0.75)'])
    
    category_counts = nse_categories.value_counts().sort_index()
    
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
    category_counts.plot(kind='bar', color=colors, edgecolor='black', alpha=0.8)
    
    plt.xlabel('Performance Category', fontsize=14)
    plt.ylabel('Number of Basins', fontsize=14)
    plt.title('Basin Performance Categories (NSE)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n? Plots saved to {output_dir}")


def main(args):
    """Main evaluation function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint['args']
    
    print(f"? Checkpoint loaded")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Training loss: {checkpoint['train_loss']:.6f}")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    print(f"  Validation NSE: {checkpoint['val_nse']:.6f}")
    
    # Load basin list
    print("\nLoading basin list...")
    test_basins = get_basin_list(args.test_basin_file)
    print(f"? Test basins: {len(test_basins)}")
    
    # Create dataset: Use LAST 5 years of test period (days 1825-3649)
    print("\nCreating test dataset...")
    test_dataset = SpatialHydroDataset(
        basin_list=test_basins,
        data_dir=args.data_dir,
        seq_length=train_args['seq_length'],
        train=False,
        mask_channel=train_args['mask_channel'],
        time_range=(1825, 3650)  # Last 5 years for final testing
    )
    print(f"? Dataset created with {len(test_dataset)} basins")
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load normalizer
    print("\nLoading normalizer...")
    normalizer = Normalizer()
    normalizer_path = Path(args.checkpoint).parent / 'normalizer.npz'
    normalizer.load(normalizer_path)
    
    # Create model
    print("\nCreating model...")
    model = SpatialRunoffModel(
        input_channels=32,
        hidden_dims=train_args['hidden_dims'],
        kernel_sizes=train_args['kernel_sizes'],
        mlp_hidden_dims=train_args['mlp_hidden_dims'],
        dropout=train_args['dropout'],
        use_mask=train_args.get('use_mask', False)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"? Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    results_df, metrics_df, overall_metrics = evaluate_model(
        model, test_loader, normalizer, device, output_dir
    )
    
    # Save results
    print("\nSaving results...")
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    metrics_df.to_csv(output_dir / 'basin_metrics.csv', index=False)
    print(f"? Saved predictions.csv")
    print(f"? Saved basin_metrics.csv")
    
    # Print results
    print("\n" + "="*70)
    print("OVERALL METRICS (All Basins Combined)")
    print("="*70)
    
    for metric, value in overall_metrics.items():
        if metric != 'count':
            print(f"  {metric.upper():12s}: {value:.6f}")
        else:
            print(f"  {metric.upper():12s}: {value}")
    
    print("\n" + "="*70)
    print("PER-BASIN METRICS (Summary Statistics)")
    print("="*70)
    
    summary_stats = metrics_df[['nse', 'rmse', 'mae', 'r_squared']].describe()
    print(summary_stats.to_string())
    
    print("\n" + "="*70)
    print("MEDIAN PER-BASIN METRICS")
    print("="*70)
    
    median_stats = metrics_df[['nse', 'rmse', 'mae', 'r_squared']].median()
    for metric, value in median_stats.items():
        print(f"  {metric.upper():12s}: {value:.6f}")
    
    # Performance breakdown
    print("\n" + "="*70)
    print("PERFORMANCE BREAKDOWN")
    print("="*70)
    
    nse_categories = {
        'Very Good (NSE > 0.75)': (metrics_df['nse'] > 0.75).sum(),
        'Good (0.65 < NSE = 0.75)': ((metrics_df['nse'] > 0.65) & (metrics_df['nse'] <= 0.75)).sum(),
        'Satisfactory (0.5 < NSE = 0.65)': ((metrics_df['nse'] > 0.5) & (metrics_df['nse'] <= 0.65)).sum(),
        'Unsatisfactory (0 < NSE = 0.5)': ((metrics_df['nse'] > 0) & (metrics_df['nse'] <= 0.5)).sum(),
        'Poor (NSE = 0)': (metrics_df['nse'] <= 0).sum()
    }
    
    for category, count in nse_categories.items():
        percentage = 100 * count / len(metrics_df)
        print(f"  {category:35s}: {count:4d} basins ({percentage:5.1f}%)")
    
    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("OVERALL METRICS (All Basins Combined)\n")
        f.write("="*70 + "\n")
        for metric, value in overall_metrics.items():
            if metric != 'count':
                f.write(f"{metric.upper():12s}: {value:.6f}\n")
            else:
                f.write(f"{metric.upper():12s}: {value}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("PER-BASIN METRICS (Summary Statistics)\n")
        f.write("="*70 + "\n")
        f.write(summary_stats.to_string())
        
        f.write("\n\n" + "="*70 + "\n")
        f.write("MEDIAN PER-BASIN METRICS\n")
        f.write("="*70 + "\n")
        for metric, value in median_stats.items():
            f.write(f"{metric.upper():12s}: {value:.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("PERFORMANCE BREAKDOWN\n")
        f.write("="*70 + "\n")
        for category, count in nse_categories.items():
            percentage = 100 * count / len(metrics_df)
            f.write(f"{category:35s}: {count:4d} basins ({percentage:5.1f}%)\n")
    
    print(f"\n? Saved summary.txt")
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(results_df, metrics_df, output_dir)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - predictions.csv: All individual predictions")
    print(f"  - basin_metrics.csv: Per-basin metrics")
    print(f"  - summary.txt: Summary statistics")
    print(f"  - *.png: Visualization plots")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained ConvLSTM model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best_model.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base data directory')
    parser.add_argument('--test_basin_file', type=str, required=True,
                        help='File containing test basin IDs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)