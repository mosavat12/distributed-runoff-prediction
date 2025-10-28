#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import glob

def fill_nan_with_neighbor(data_2d):
    """
    Replace NaN values with nearest non-NaN neighbor value
    Simple and fast approach for sparse NaNs
    
    Parameters:
    data_2d: 2D array (61, 61)
    
    Returns:
    filled array
    """
    filled = data_2d.copy()
    nan_mask = np.isnan(filled)
    
    if not np.any(nan_mask):
        return filled
    
    # Get indices of NaN pixels
    nan_indices = np.argwhere(nan_mask)
    
    # For each NaN pixel, find nearest non-NaN neighbor
    for idx in nan_indices:
        i, j = idx
        
        # Check neighbors in expanding radius
        for radius in range(1, 32):
            found = False
            
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = i + di, j + dj
                    
                    if 0 <= ni < 61 and 0 <= nj < 61:
                        if not np.isnan(filled[ni, nj]):
                            filled[i, j] = filled[ni, nj]
                            found = True
                            break
                
                if found:
                    break
            
            if found:
                break
    
    return filled

def process_single_basin(basin_index):
    """
    Fix DEM and MODIS variables for a single basin using memory mapping
    
    Parameters:
    basin_index: SLURM array task ID (1-based indexing)
    
    Returns:
    0 for success, 1 for failure
    """
    try:
        final_inputs_dir = "/icebox/data/shares/mh2/mosavat/Distributed/train_final_inputs"
        static_base = "/icebox/data/shares/mh2/mosavat/Distributed/Static_variables"
        
        # Get sorted list of all basin files
        npy_files = sorted(glob.glob(os.path.join(final_inputs_dir, "*.npy")))
        
        if len(npy_files) == 0:
            print("ERROR: No .npy files found in final inputs directory!")
            return 1
        
        # Convert to 0-based index
        file_index = basin_index - 1
        
        if file_index < 0 or file_index >= len(npy_files):
            print(f"ERROR: Basin index {basin_index} out of range (1-{len(npy_files)})")
            return 1
        
        filepath = npy_files[file_index]
        basin_id = os.path.basename(filepath).replace('.npy', '')
        
        print(f"PROCESSING: {basin_id} (index {basin_index}/{len(npy_files)})")
        
        # Use memory mapping - much faster for large files!
        data = np.load(filepath, mmap_mode='r+')
        
        if data.shape != (61, 61, 32, 3650):
            print(f"ERROR: Unexpected shape {data.shape} for {basin_id}")
            return 1
        
        # 1. Fix DEM Elevation and Slope (indices 7-8)
        print(f"  Fixing DEM Elevation and Slope...")
        dem_elev_slop_path = os.path.join(static_base, "DEM/Elev_Slop/files", f"{basin_id}.npy")
        
        if not os.path.exists(dem_elev_slop_path):
            print(f"ERROR: DEM file not found: {dem_elev_slop_path}")
            return 1
        
        dem_elev_slop = np.load(dem_elev_slop_path)
        
        if dem_elev_slop.shape != (61, 61, 2):
            print(f"ERROR: Unexpected DEM shape {dem_elev_slop.shape}")
            return 1
        
        # Update elevation - write to each timestep individually (faster than broadcasting)
        elev_value = dem_elev_slop[:, :, 0]
        for t in range(3650):
            data[:, :, 7, t] = elev_value
        
        # Update slope
        slope_value = dem_elev_slop[:, :, 1]
        for t in range(3650):
            data[:, :, 8, t] = slope_value
        
        # 2. Fix MODIS variables (indices 18-22)
        print(f"  Fixing MODIS NaN values...")
        for k in range(5):
            modis_idx = 18 + k
            
            # Get the 2D slice from first timestep
            modis_2d = data[:, :, modis_idx, 0].copy()
            
            # Check if there are any NaNs
            if np.any(np.isnan(modis_2d)):
                n_nans = np.sum(np.isnan(modis_2d))
                print(f"    MODIS variable {k}: Found {n_nans} NaN pixels, filling...")
                
                # Fill NaNs
                filled = fill_nan_with_neighbor(modis_2d)
                
                # Update across all timesteps
                for t in range(3650):
                    data[:, :, modis_idx, t] = filled
        
        # Flush changes to disk
        data.flush()
        
        print(f"SUCCESS: {basin_id} - Fixed and saved")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python fix_dem_modis.py <basin_index>")
        print("Example: python fix_dem_modis.py 1")
        sys.exit(1)
    
    try:
        basin_index = int(sys.argv[1])
    except ValueError:
        print("ERROR: Basin index must be an integer")
        sys.exit(1)
    
    exit_code = process_single_basin(basin_index)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()