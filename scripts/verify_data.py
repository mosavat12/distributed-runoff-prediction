"""
Data Verification and Preparation Script

This script helps you:
1. Verify data structure and formats
2. Create basin list files
3. Check for missing or corrupted files
4. Generate data statistics
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def verify_npy_file(filepath, expected_shape=(61, 61, 32, 3650)):
    """
    Verify a single .npy file.
    
    Returns:
        dict: Verification results
    """
    try:
        data = np.load(filepath)
        
        issues = []
        
        # Check shape
        if data.shape != expected_shape:
            issues.append(f"Wrong shape: {data.shape} (expected {expected_shape})")
        
        # Check for NaN
        nan_count = np.isnan(data).sum()
        if nan_count > 0:
            issues.append(f"Contains {nan_count} NaN values")
        
        # Check for Inf
        inf_count = np.isinf(data).sum()
        if inf_count > 0:
            issues.append(f"Contains {inf_count} Inf values")
        
        # Check mask channel
        mask = data[:, :, 31, 0]
        valid_pixels = (mask > 0).sum()
        if valid_pixels == 0:
            issues.append("Mask channel (31) is all zeros")
        
        return {
            'success': len(issues) == 0,
            'shape': data.shape,
            'size_mb': filepath.stat().st_size / (1024**2),
            'nan_count': nan_count,
            'inf_count': inf_count,
            'valid_pixels': valid_pixels,
            'issues': issues
        }
    
    except Exception as e:
        return {
            'success': False,
            'issues': [f"Error loading file: {str(e)}"]
        }


def verify_csv_file(filepath, expected_length=3650):
    """
    Verify a single .csv target file.
    
    Returns:
        dict: Verification results
    """
    try:
        df = pd.read_csv(filepath)
        
        issues = []
        
        # Check columns
        if 'runoff' not in df.columns:
            issues.append("Missing 'runoff' column")
        
        if 'datetime' not in df.columns:
            issues.append("Missing 'datetime' column")
        
        # Check length
        if len(df) != expected_length:
            issues.append(f"Wrong length: {len(df)} (expected {expected_length})")
        
        # Check for NaN in runoff
        if 'runoff' in df.columns:
            nan_count = df['runoff'].isna().sum()
            if nan_count > 0:
                issues.append(f"Contains {nan_count} NaN runoff values")
        
        return {
            'success': len(issues) == 0,
            'length': len(df),
            'columns': list(df.columns),
            'runoff_range': [df['runoff'].min(), df['runoff'].max()] if 'runoff' in df.columns else None,
            'issues': issues
        }
    
    except Exception as e:
        return {
            'success': False,
            'issues': [f"Error loading file: {str(e)}"]
        }


def create_basin_lists(data_dir):
    """
    Create basin list files by scanning input directories.
    """
    data_dir = Path(data_dir)
    
    # Get training basins
    train_input_dir = data_dir / 'train_final_inputs'
    train_basins = sorted([f.stem for f in train_input_dir.glob('*.npy')])
    
    # Get test basins
    test_input_dir = data_dir / 'test_final_inputs'
    test_basins = sorted([f.stem for f in test_input_dir.glob('*.npy')])
    
    # Save basin lists
    train_file = data_dir / 'train_basins.txt'
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_basins))
    
    test_file = data_dir / 'test_basins.txt'
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_basins))
    
    print(f"Created basin lists:")
    print(f"  {train_file} ({len(train_basins)} basins)")
    print(f"  {test_file} ({len(test_basins)} basins)")
    
    return train_basins, test_basins


def verify_dataset(data_dir, basin_list, train=True, max_check=None):
    """
    Verify entire dataset.
    
    Args:
        data_dir: Base data directory
        basin_list: List of basin IDs to check
        train: If True, check training data; else test data
        max_check: Maximum number of basins to check (None = all)
    """
    data_dir = Path(data_dir)
    
    if train:
        input_dir = data_dir / 'train_final_inputs'
        target_dir = data_dir / 'train_targets'
        split = 'TRAINING'
    else:
        input_dir = data_dir / 'test_final_inputs'
        target_dir = data_dir / 'test_targets'
        split = 'TEST'
    
    print(f"\n{'='*60}")
    print(f"VERIFYING {split} DATA")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Basins to check: {len(basin_list)}")
    
    if max_check and max_check < len(basin_list):
        basin_list = basin_list[:max_check]
        print(f"Limiting to first {max_check} basins")
    
    results = {
        'total': len(basin_list),
        'input_success': 0,
        'target_success': 0,
        'both_success': 0,
        'input_issues': [],
        'target_issues': [],
        'missing_input': [],
        'missing_target': []
    }
    
    print("\nChecking files...")
    for basin_id in tqdm(basin_list):
        # Check input file
        input_file = input_dir / f"{basin_id}.npy"
        if input_file.exists():
            input_result = verify_npy_file(input_file)
            if input_result['success']:
                results['input_success'] += 1
            else:
                results['input_issues'].append({
                    'basin_id': basin_id,
                    'issues': input_result['issues']
                })
        else:
            results['missing_input'].append(basin_id)
        
        # Check target file
        target_file = target_dir / f"{basin_id}.csv"
        if target_file.exists():
            target_result = verify_csv_file(target_file)
            if target_result['success']:
                results['target_success'] += 1
            else:
                results['target_issues'].append({
                    'basin_id': basin_id,
                    'issues': target_result['issues']
                })
        else:
            results['missing_target'].append(basin_id)
        
        # Check if both successful
        if input_file.exists() and target_file.exists():
            if input_result['success'] and target_result['success']:
                results['both_success'] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY - {split}")
    print(f"{'='*60}")
    print(f"Total basins checked: {results['total']}")
    print(f"Input files OK: {results['input_success']}/{results['total']} ({100*results['input_success']/results['total']:.1f}%)")
    print(f"Target files OK: {results['target_success']}/{results['total']} ({100*results['target_success']/results['total']:.1f}%)")
    print(f"Both files OK: {results['both_success']}/{results['total']} ({100*results['both_success']/results['total']:.1f}%)")
    
    if results['missing_input']:
        print(f"\nMISSING INPUT FILES: {len(results['missing_input'])}")
        for basin_id in results['missing_input'][:10]:
            print(f"  - {basin_id}")
        if len(results['missing_input']) > 10:
            print(f"  ... and {len(results['missing_input']) - 10} more")
    
    if results['missing_target']:
        print(f"\nMISSING TARGET FILES: {len(results['missing_target'])}")
        for basin_id in results['missing_target'][:10]:
            print(f"  - {basin_id}")
        if len(results['missing_target']) > 10:
            print(f"  ... and {len(results['missing_target']) - 10} more")
    
    if results['input_issues']:
        print(f"\nINPUT FILE ISSUES: {len(results['input_issues'])}")
        for item in results['input_issues'][:5]:
            print(f"  Basin {item['basin_id']}:")
            for issue in item['issues']:
                print(f"    - {issue}")
        if len(results['input_issues']) > 5:
            print(f"  ... and {len(results['input_issues']) - 5} more basins with issues")
    
    if results['target_issues']:
        print(f"\nTARGET FILE ISSUES: {len(results['target_issues'])}")
        for item in results['target_issues'][:5]:
            print(f"  Basin {item['basin_id']}:")
            for issue in item['issues']:
                print(f"    - {issue}")
        if len(results['target_issues']) > 5:
            print(f"  ... and {len(results['target_issues']) - 5} more basins with issues")
    
    return results


def main(args):
    """Main function."""
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return
    
    # Step 1: Create basin lists if requested
    if args.create_lists:
        print("Creating basin list files...")
        train_basins, test_basins = create_basin_lists(data_dir)
    else:
        # Load existing basin lists
        train_file = data_dir / 'train_basins.txt'
        test_file = data_dir / 'test_basins.txt'
        
        if not train_file.exists() or not test_file.exists():
            print("Basin list files not found. Creating them...")
            train_basins, test_basins = create_basin_lists(data_dir)
        else:
            with open(train_file, 'r') as f:
                train_basins = [line.strip() for line in f if line.strip()]
            with open(test_file, 'r') as f:
                test_basins = [line.strip() for line in f if line.strip()]
    
    # Step 2: Verify training data
    if args.verify_train:
        train_results = verify_dataset(
            data_dir, train_basins, train=True, max_check=args.max_check
        )
    
    # Step 3: Verify test data
    if args.verify_test:
        test_results = verify_dataset(
            data_dir, test_basins, train=False, max_check=args.max_check
        )
    
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Fix any issues identified above")
    print("2. Run test data loading with: python spatial_dataset.py")
    print("3. Submit training job with: sbatch train_convlstm.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify and prepare data for ConvLSTM training')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base data directory')
    parser.add_argument('--create_lists', action='store_true',
                        help='Create basin list files')
    parser.add_argument('--verify_train', action='store_true',
                        help='Verify training data')
    parser.add_argument('--verify_test', action='store_true',
                        help='Verify test data')
    parser.add_argument('--max_check', type=int, default=None,
                        help='Maximum number of basins to check per split')
    
    args = parser.parse_args()
    
    # If no verification flags set, verify both
    if not args.verify_train and not args.verify_test:
        args.verify_train = True
        args.verify_test = True
    
    main(args)