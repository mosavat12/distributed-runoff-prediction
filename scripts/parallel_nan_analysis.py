#!/usr/bin/env python3
"""
NaN Detection and Analysis for ConvLSTM Spatial Data
Analyzes train_final_inputs and test_final_inputs directories
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import json

class SpatialNaNAnalyzer:
    """Analyze NaN values in ConvLSTM spatial input data"""
    
    def __init__(self, base_dir: str):
        """
        Initialize analyzer
        
        Args:
            base_dir: Base directory containing train_final_inputs, test_final_inputs, etc.
        """
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / "train_final_inputs"
        self.test_dir = self.base_dir / "test_final_inputs"
        self.train_basins_file = self.base_dir / "train_basins.txt"
        self.test_basins_file = self.base_dir / "test_basins.txt"
        
        # Variable names - 32 channels total
        self.variable_names = self._get_variable_names()
        self.results = {'train': {}, 'test': {}}
        
    def _get_variable_names(self) -> List[str]:
        """Get the 32 variable names for your channels"""
        # 6 dynamic variables (repeated across time)
        dynamic_vars = ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'pet']
        
        # 26 static variables (constant across time)
        static_vars = [
            'elev_mean', 'slope_mean', 'area_gages2', 
            'frac_forest', 'lai_max', 'lai_diff', 
            'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 
            'dom_land_cover', 'root_depth_50', 'root_depth_99',
            'soil_depth_pelletier', 'soil_depth_statsgo', 
            'soil_porosity', 'soil_conductivity', 
            'max_water_content', 'sand_frac', 'silt_frac', 
            'clay_frac', 'water_frac', 'organic_frac', 
            'other_frac', 'carbonate_rocks_frac', 
            'geol_permeability', 'p_mean'
        ]
        
        # Basin mask as channel 31 (index 31)
        return dynamic_vars + static_vars + ['basin_mask']
    
    def load_basin_list(self, basin_file: Path) -> List[str]:
        """Load basin IDs from text file"""
        if not basin_file.exists():
            print(f"Warning: Basin file not found: {basin_file}")
            return []
        
        with open(basin_file, 'r') as f:
            basins = [line.strip() for line in f if line.strip()]
        return basins
    
    def analyze_basin(self, basin_id: str, data_type: str = 'train') -> Dict:
        """
        Analyze a single basin's .npy file
        
        Args:
            basin_id: Basin identifier
            data_type: 'train' or 'test'
            
        Returns:
            Dictionary with NaN statistics
        """
        # Determine file path
        if data_type == 'train':
            filepath = self.train_dir / f"{basin_id}.npy"
        else:
            filepath = self.test_dir / f"{basin_id}.npy"
        
        if not filepath.exists():
            return {
                'basin_id': basin_id,
                'error': f'File not found: {filepath}'
            }
        
        try:
            # Load data - expected shape: (61, 61, 32, 3650)
            data = np.load(filepath)
            
            # Basic statistics
            total_elements = data.size
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            
            results = {
                'basin_id': basin_id,
                'filepath': str(filepath),
                'shape': data.shape,
                'total_elements': total_elements,
                'nan_count': int(nan_count),
                'nan_percentage': float((nan_count / total_elements) * 100),
                'inf_count': int(inf_count),
                'inf_percentage': float((inf_count / total_elements) * 100),
                'per_variable': {}
            }
            
            # Analyze each variable/channel
            for ch_idx in range(32):
                var_name = self.variable_names[ch_idx] if ch_idx < len(self.variable_names) else f'channel_{ch_idx}'
                
                # Extract channel data: (61, 61, 3650)
                channel_data = data[:, :, ch_idx, :]
                
                # Count NaN and Inf
                ch_nan_count = np.isnan(channel_data).sum()
                ch_inf_count = np.isinf(channel_data).sum()
                
                var_stats = {
                    'nan_count': int(ch_nan_count),
                    'nan_percentage': float((ch_nan_count / channel_data.size) * 100),
                    'inf_count': int(ch_inf_count),
                    'inf_percentage': float((ch_inf_count / channel_data.size) * 100)
                }
                
                # Get valid data statistics
                valid_mask = ~np.isnan(channel_data) & ~np.isinf(channel_data)
                if valid_mask.any():
                    valid_data = channel_data[valid_mask]
                    var_stats.update({
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max()),
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std())
                    })
                    
                    # Check if constant (e.g., static variables or mask)
                    if var_stats['std'] < 1e-8:
                        var_stats['is_constant'] = True
                        var_stats['constant_value'] = float(valid_data[0])
                
                # Spatial NaN pattern - which pixels have NaN
                if ch_nan_count > 0:
                    # Check which spatial locations have any NaN across time
                    spatial_nan = np.isnan(channel_data).any(axis=2)
                    var_stats['spatial_nan_pixels'] = int(spatial_nan.sum())
                    var_stats['spatial_nan_percentage'] = float((spatial_nan.sum() / (61*61)) * 100)
                    
                    # Check temporal pattern - which timesteps have NaN
                    temporal_nan = np.isnan(channel_data).any(axis=(0, 1))
                    var_stats['temporal_nan_timesteps'] = int(temporal_nan.sum())
                    var_stats['temporal_nan_percentage'] = float((temporal_nan.sum() / 3650) * 100)
                
                results['per_variable'][var_name] = var_stats
            
            return results
            
        except Exception as e:
            return {
                'basin_id': basin_id,
                'error': str(e)
            }
    
    def analyze_dataset(self, data_type: str = 'train', max_basins: int = None) -> None:
        """
        Analyze all basins in train or test set
        
        Args:
            data_type: 'train' or 'test'
            max_basins: Maximum number of basins to analyze
        """
        # Load basin list
        basin_file = self.train_basins_file if data_type == 'train' else self.test_basins_file
        basins = self.load_basin_list(basin_file)
        
        if not basins:
            # Try to find basins from directory
            data_dir = self.train_dir if data_type == 'train' else self.test_dir
            if data_dir.exists():
                npy_files = list(data_dir.glob("*.npy"))
                basins = [f.stem for f in npy_files]
                print(f"Found {len(basins)} .npy files in {data_dir}")
            else:
                print(f"Error: Directory not found: {data_dir}")
                return
        
        if max_basins:
            basins = basins[:max_basins]
        
        print(f"\nAnalyzing {len(basins)} {data_type} basins...")
        
        for basin_id in tqdm(basins, desc=f"Analyzing {data_type} data"):
            self.results[data_type][basin_id] = self.analyze_basin(basin_id, data_type)
    
    def get_summary(self, data_type: str = 'train') -> Dict:
        """Generate summary statistics for train or test data"""
        
        data_results = self.results[data_type]
        
        if not data_results:
            return {
                'data_type': data_type,
                'total_basins': 0,
                'warning': f'No {data_type} basins analyzed'
            }
        
        summary = {
            'data_type': data_type,
            'total_basins': len(data_results),
            'basins_with_nan': 0,
            'basins_with_inf': 0,
            'basins_with_errors': 0,
            'variables_with_issues': {},
            'constant_variables': set()
        }
        
        all_nan_percentages = []
        
        for basin_id, result in data_results.items():
            if 'error' in result:
                summary['basins_with_errors'] += 1
                continue
            
            if result['nan_count'] > 0:
                summary['basins_with_nan'] += 1
                all_nan_percentages.append(result['nan_percentage'])
            
            if result.get('inf_count', 0) > 0:
                summary['basins_with_inf'] += 1
            
            # Track variables with issues
            for var_name, var_stats in result.get('per_variable', {}).items():
                if var_stats['nan_count'] > 0 or var_stats['inf_count'] > 0:
                    if var_name not in summary['variables_with_issues']:
                        summary['variables_with_issues'][var_name] = {
                            'basins_affected': [],
                            'total_nan_count': 0,
                            'total_inf_count': 0,
                            'mean_nan_percentage': []
                        }
                    
                    summary['variables_with_issues'][var_name]['basins_affected'].append(basin_id)
                    summary['variables_with_issues'][var_name]['total_nan_count'] += var_stats['nan_count']
                    summary['variables_with_issues'][var_name]['total_inf_count'] += var_stats['inf_count']
                    summary['variables_with_issues'][var_name]['mean_nan_percentage'].append(var_stats['nan_percentage'])
                
                # Track constant variables
                if var_stats.get('is_constant', False):
                    summary['constant_variables'].add(var_name)
        
        # Calculate final statistics
        if all_nan_percentages:
            summary['nan_statistics'] = {
                'min_percentage': min(all_nan_percentages),
                'max_percentage': max(all_nan_percentages),
                'mean_percentage': np.mean(all_nan_percentages)
            }
        
        # Finalize variable statistics
        for var_name in summary['variables_with_issues']:
            var_data = summary['variables_with_issues'][var_name]
            var_data['num_basins_affected'] = len(var_data['basins_affected'])
            var_data['mean_nan_percentage'] = float(np.mean(var_data['mean_nan_percentage']))
            
            # Keep only first 5 basin IDs to avoid clutter
            if len(var_data['basins_affected']) > 5:
                var_data['basins_affected'] = var_data['basins_affected'][:5] + [f"...and {len(var_data['basins_affected'])-5} more"]
        
        summary['constant_variables'] = list(summary['constant_variables'])
        
        return summary
    
    def print_report(self, detailed: bool = False) -> None:
        """Print comprehensive report for both train and test data"""
        
        print("\n" + "="*80)
        print("SPATIAL DATA NaN ANALYSIS REPORT")
        print("="*80)
        print(f"Base directory: {self.base_dir}")
        
        for data_type in ['train', 'test']:
            if not self.results[data_type]:
                continue
            
            summary = self.get_summary(data_type)
            
            print(f"\n{'-'*40}")
            print(f"{data_type.upper()} DATA ANALYSIS")
            print(f"{'-'*40}")
            
            if 'warning' in summary:
                print(f"?? {summary['warning']}")
                continue
            
            print(f"Total basins analyzed: {summary['total_basins']}")
            print(f"Basins with NaN: {summary['basins_with_nan']} ({summary['basins_with_nan']/summary['total_basins']*100:.1f}%)")
            print(f"Basins with Inf: {summary['basins_with_inf']} ({summary['basins_with_inf']/summary['total_basins']*100:.1f}%)")
            
            if 'nan_statistics' in summary:
                print(f"\nNaN Statistics:")
                print(f"  Min: {summary['nan_statistics']['min_percentage']:.6f}%")
                print(f"  Max: {summary['nan_statistics']['max_percentage']:.6f}%")
                print(f"  Mean: {summary['nan_statistics']['mean_percentage']:.6f}%")
            
            if summary['variables_with_issues']:
                print(f"\nVariables with NaN/Inf values:")
                print(f"{'Variable':<20} {'Affected Basins':<15} {'Mean NaN %':<12}")
                print("-"*50)
                
                sorted_vars = sorted(summary['variables_with_issues'].items(),
                                   key=lambda x: x[1]['num_basins_affected'],
                                   reverse=True)
                
                for var_name, stats in sorted_vars[:10]:  # Show top 10
                    print(f"{var_name:<20} {stats['num_basins_affected']:<15} {stats['mean_nan_percentage']:.4f}%")
            
            if summary['constant_variables']:
                print(f"\nConstant variables (zero variance):")
                for var in sorted(summary['constant_variables']):
                    print(f"  - {var}")
            
            # Detailed basin-level report
            if detailed:
                print(f"\n{data_type.upper()} - Basin Details (showing basins with issues):")
                print("-"*60)
                
                count = 0
                for basin_id, result in self.results[data_type].items():
                    if 'error' in result:
                        print(f"{basin_id}: ERROR - {result['error']}")
                        count += 1
                    elif result.get('nan_count', 0) > 0 or result.get('inf_count', 0) > 0:
                        print(f"{basin_id}:")
                        print(f"  NaN: {result['nan_count']:,} ({result['nan_percentage']:.4f}%)")
                        if result.get('inf_count', 0) > 0:
                            print(f"  Inf: {result['inf_count']:,} ({result['inf_percentage']:.4f}%)")
                        
                        # Show top variables with issues
                        var_issues = [(name, stats['nan_percentage']) 
                                    for name, stats in result['per_variable'].items()
                                    if stats['nan_count'] > 0]
                        if var_issues:
                            var_issues.sort(key=lambda x: x[1], reverse=True)
                            print("  Top variables with NaN:")
                            for var_name, nan_pct in var_issues[:3]:
                                print(f"    - {var_name}: {nan_pct:.4f}%")
                        count += 1
                    
                    if count >= 10:  # Limit detailed output
                        print("  ...showing first 10 basins with issues")
                        break
    
    def save_results(self, output_file: str) -> None:
        """Save results to JSON file"""
        output = {
            'base_directory': str(self.base_dir),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'train_summary': self.get_summary('train'),
            'test_summary': self.get_summary('test'),
            'train_results': self.results['train'],
            'test_results': self.results['test']
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze NaN values in ConvLSTM spatial data')
    parser.add_argument('--base_dir', type=str, 
                       default='/icebox/data/shares/mh2/mosavat/Distributed',
                       help='Base directory containing train_final_inputs, test_final_inputs, etc.')
    parser.add_argument('--train_only', action='store_true',
                       help='Analyze only training data')
    parser.add_argument('--test_only', action='store_true',
                       help='Analyze only test data')
    parser.add_argument('--max_basins', type=int, default=None,
                       help='Maximum number of basins to analyze per dataset')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed basin-level information')
    parser.add_argument('--save', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SpatialNaNAnalyzer(args.base_dir)
    
    # Analyze datasets
    if not args.test_only:
        analyzer.analyze_dataset('train', max_basins=args.max_basins)
    
    if not args.train_only:
        analyzer.analyze_dataset('test', max_basins=args.max_basins)
    
    # Print report
    analyzer.print_report(detailed=args.detailed)
    
    # Save if requested
    if args.save:
        analyzer.save_results(args.save)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()