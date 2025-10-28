#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import glob
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

def priestley_taylor_pet(tmin, tmax, srad):
    """
    Priestley-Taylor PET calculation with NaN handling
    
    Parameters:
    tmin: minimum daily temperature (degrees C) - shape (n_days,)
    tmax: maximum daily temperature (degrees C) - shape (n_days,)
    srad: shortwave radiation (W/m2) - shape (n_days,)
    
    Returns:
    pet: Potential evapotranspiration (mm/day) - shape (n_days,)
    """
    # Handle NaN values
    mask = np.isnan(tmin) | np.isnan(tmax) | np.isnan(srad)
    
    # Calculate mean temperature
    tmean = (tmin + tmax) / 2.0
    
    # Saturation vapor pressure (kPa)
    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    
    # Psychrometric constant (kPa/degrees C)
    gamma = 0.665
    
    # Slope of saturation vapor pressure curve (kPa/degrees C)
    delta = 4098 * es / ((tmean + 237.3) ** 2)
    
    # Convert solar radiation from W/m2 to MJ/m2/day
    srad_mj = srad * 0.0864
    
    # Net radiation (MJ/m2/day) - simplified as 77% of solar radiation
    rn = 0.77 * srad_mj
    
    # Soil heat flux (assumed to be 0 for daily calculations)
    g = 0
    
    # Priestley-Taylor coefficient
    alpha = 1.26
    
    # Calculate PET (mm/day)
    pet = alpha * (delta / (delta + gamma)) * (rn - g) / 2.45
    
    # Ensure PET is non-negative
    pet = np.maximum(pet, 0)
    
    # Set NaN values where inputs were NaN
    pet[mask] = np.nan
    
    return pet

def seasonality_metric(prcp, doy):
    """Calculate precipitation seasonality using sine curve fitting"""
    try:
        def objective(phi):
            x = 2 * np.pi * (doy - phi) / 365
            A = np.column_stack([np.sin(x), np.cos(x), np.ones(len(doy))])
            try:
                coeffs = np.linalg.lstsq(A, prcp, rcond=None)[0]
                fitted = A @ coeffs
                return np.sum((prcp - fitted)**2)
            except:
                return 1e10
        
        result = minimize_scalar(objective, bounds=(1, 365), method='bounded')
        phi_opt = result.x
        
        x_opt = 2 * np.pi * (doy - phi_opt) / 365
        A_opt = np.column_stack([np.sin(x_opt), np.cos(x_opt), np.ones(len(doy))])
        coeffs = np.linalg.lstsq(A_opt, prcp, rcond=None)[0]
        
        summer_phase = 182.5
        phase_diff = abs(phi_opt - summer_phase)
        if phase_diff > 182.5:
            phase_diff = 365 - phase_diff
        
        seasonality = (coeffs[0]**2 + coeffs[1]**2)**0.5 / np.mean(prcp)
        if phase_diff > 91.25:
            seasonality *= -1
        
        return seasonality
    except:
        return np.nan

def avg_event_duration(binary_array):
    """Calculate average duration of consecutive True events"""
    try:
        if not np.any(binary_array) or len(binary_array) < 2:
            return 0
        
        diff = np.diff(np.concatenate([[False], binary_array, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0 or len(ends) == 0:
            return 0
        
        durations = ends - starts
        return np.mean(durations) if len(durations) > 0 else 0
    except:
        return np.nan

def calculate_pixel_characteristics(prcp, tmin, tmax, pet, doy):
    """
    Calculate 9 static characteristics for a single pixel's timeseries
    
    Parameters:
    prcp: precipitation timeseries (n_days,)
    tmin: minimum temperature timeseries (n_days,)
    tmax: maximum temperature timeseries (n_days,)
    pet: PET timeseries (n_days,)
    doy: day of year array (n_days,)
    
    Returns:
    results: array of 9 static variables
    """
    results = np.full(9, np.nan)
    
    # 1. Mean daily precipitation
    if not np.all(np.isnan(prcp)):
        results[0] = np.nanmean(prcp)
    
    # 2. Mean daily PET
    if not np.all(np.isnan(pet)):
        results[1] = np.nanmean(pet)
    
    # 3. Aridity (PET/P)
    if not np.isnan(results[0]) and not np.isnan(results[1]) and results[0] > 0:
        results[2] = results[1] / results[0]
    
    # 4. Precipitation seasonality
    valid_prcp = ~np.isnan(prcp)
    if np.sum(valid_prcp) > 100:
        try:
            results[3] = seasonality_metric(prcp[valid_prcp], doy[valid_prcp])
        except:
            pass
    
    # 5. Fraction of snow (precipitation on days with tmin < 0C)
    valid_snow = ~np.isnan(prcp) & ~np.isnan(tmin) & (prcp > 0)
    if np.sum(valid_snow) > 0:
        snow_days = valid_snow & (tmin < 0)
        results[4] = np.sum(prcp[snow_days]) / np.sum(prcp[valid_snow])
    
    # 6. High precipitation frequency
    if not np.isnan(results[0]) and results[0] > 0:
        high_thresh = 5 * results[0]
        valid_prcp_days = ~np.isnan(prcp)
        if np.sum(valid_prcp_days) > 0:
            high_prec_days = (prcp >= high_thresh) & valid_prcp_days
            results[5] = np.sum(high_prec_days) / np.sum(valid_prcp_days)
    
    # 7. High precipitation duration
    if not np.isnan(results[0]) and results[0] > 0:
        high_thresh = 5 * results[0]
        valid_prcp_days = ~np.isnan(prcp)
        if np.sum(valid_prcp_days) > 100:
            high_prec_binary = (prcp >= high_thresh) & valid_prcp_days
            results[6] = avg_event_duration(high_prec_binary)
    
    # 8. Low precipitation frequency
    valid_prcp_days = ~np.isnan(prcp)
    if np.sum(valid_prcp_days) > 0:
        low_prec_days = (prcp < 1.0) & valid_prcp_days
        results[7] = np.sum(low_prec_days) / np.sum(valid_prcp_days)
    
    # 9. Low precipitation duration
    valid_prcp_days = ~np.isnan(prcp)
    if np.sum(valid_prcp_days) > 100:
        low_prec_binary = (prcp < 1.0) & valid_prcp_days
        results[8] = avg_event_duration(low_prec_binary)
    
    return results

def load_static_variable(base_dir, basin_id, expected_shape):
    """
    Load static variable file for a given basin
    
    Parameters:
    base_dir: directory containing the static variable files
    basin_id: basin identifier (filename without .npy)
    expected_shape: expected shape of the array (e.g., (61, 61) or (61, 61, N))
    
    Returns:
    numpy array with the static variable(s)
    """
    filepath = os.path.join(base_dir, f"{basin_id}.npy")
    data = np.load(filepath)
    
    # Validate shape
    if data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data.shape} for {filepath}")
    
    return data

def process_single_basin(basin_index):
    """
    Process a single basin file based on array index
    
    Parameters:
    basin_index: SLURM array task ID (1-based indexing)
    
    Returns:
    0 for success, 1 for failure
    """
    try:
        input_dir = "/icebox/data/shares/mh2/mosavat/Distributed/test_inputs"
        output_dir = "/icebox/data/shares/mh2/mosavat/Distributed/test_final_inputs_REV2"
        static_base = "/icebox/data/shares/mh2/mosavat/Distributed/Static_variables"
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Get sorted list of all basin files
        npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
        
        if len(npy_files) == 0:
            print("ERROR: No .npy files found in input directory!")
            return 1
        
        # Convert to 0-based index
        file_index = basin_index - 1
        
        if file_index < 0 or file_index >= len(npy_files):
            print(f"ERROR: Basin index {basin_index} out of range (1-{len(npy_files)})")
            return 1
        
        filepath = npy_files[file_index]
        basin_id = os.path.basename(filepath).replace('.npy', '')
        
        # Check if output already exists
        output_path = os.path.join(output_dir, f"{basin_id}.npy")
        if os.path.exists(output_path):
            print(f"SKIPPED: {basin_id} - output already exists")
            return 0
        
        print(f"PROCESSING: {basin_id} (index {basin_index}/{len(npy_files)})")
        
        # Load data: (61, 61, 5, 3650)
        data = np.load(filepath)
        
        if data.shape != (61, 61, 5, 3650):
            print(f"ERROR: Unexpected shape {data.shape} for {basin_id}")
            return 1
        
        n_days = data.shape[3]
        
        # Extract variables
        prcp = data[:, :, 0, :]  # (61, 61, 3650)
        tmin = data[:, :, 1, :]
        tmax = data[:, :, 2, :]
        srad = data[:, :, 3, :]
        vp = data[:, :, 4, :]
        
        # Initialize output array: (61, 61, 32, 3650)
        output = np.zeros((61, 61, 32, n_days), dtype=np.float32)
        
        # Copy original 5 dynamic variables (indices 0-4)
        output[:, :, 0, :] = prcp
        output[:, :, 1, :] = tmin
        output[:, :, 2, :] = tmax
        output[:, :, 3, :] = srad
        output[:, :, 4, :] = vp
        
        # Create day of year array
        n_years = n_days // 365
        doy = np.tile(np.arange(1, 366), n_years)[:n_days]
        
        # Calculate PET and Daymet static variables for each pixel
        print(f"  Calculating PET and static variables for 3721 pixels...")
        for i in range(61):
            for j in range(61):
                # Calculate PET for this pixel (index 5 - dynamic)
                pet_ts = priestley_taylor_pet(tmin[i, j, :], tmax[i, j, :], srad[i, j, :])
                output[i, j, 5, :] = pet_ts
                
                # Calculate 9 Daymet static variables for this pixel (indices 9-17)
                static_vars = calculate_pixel_characteristics(
                    prcp[i, j, :], 
                    tmin[i, j, :], 
                    tmax[i, j, :], 
                    pet_ts, 
                    doy
                )
                
                # Repeat static variables across all days
                for k in range(9):
                    output[i, j, 9 + k, :] = static_vars[k]
        
        # Load static variables from external sources
        print(f"  Loading external static variables...")
        
        # 1. DEM Active_Cells (1 variable) - index 6
        dem_active = load_static_variable(
            os.path.join(static_base, "DEM/Active_Cells/files"),
            basin_id,
            (61, 61)
        )
        output[:, :, 6, :] = dem_active[:, :, np.newaxis]
        
        # 2. DEM Elev_Slop (2 variables) - indices 7-8
        dem_elev_slop = load_static_variable(
            os.path.join(static_base, "DEM/Elev_Slop/files"),
            basin_id,
            (61, 61, 2)
        )
        output[:, :, 7, :] = dem_elev_slop[:, :, 0, np.newaxis]  # elevation
        output[:, :, 8, :] = dem_elev_slop[:, :, 1, np.newaxis]  # slope
        
        # 3. MODIS (5 variables) - indices 18-22
        modis = load_static_variable(
            os.path.join(static_base, "MODIS/final_files"),
            basin_id,
            (61, 61, 5)
        )
        for k in range(5):
            output[:, :, 18 + k, :] = modis[:, :, k, np.newaxis]
        
        # 4. STATSGO (7 variables) - indices 23-29
        statsgo = load_static_variable(
            os.path.join(static_base, "STATSGO/files"),
            basin_id,
            (61, 61, 7)
        )
        for k in range(7):
            output[:, :, 23 + k, :] = statsgo[:, :, k, np.newaxis]
        
        # 5. GLHYMPS (1 variable) - index 30
        glhymps = load_static_variable(
            os.path.join(static_base, "GLHYMPS/files"),
            basin_id,
            (61, 61)
        )
        output[:, :, 30, :] = glhymps[:, :, np.newaxis]
        
        # 6. GLiM (1 variable) - index 31
        glim = load_static_variable(
            os.path.join(static_base, "GLiM/files"),
            basin_id,
            (61, 61)
        )
        output[:, :, 31, :] = glim[:, :, np.newaxis]
        
        # Save output
        print(f"  Saving output...")
        np.save(output_path, output)
        
        print(f"SUCCESS: {basin_id} - (61, 61, 32, 3650) saved")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python distributed_vars_calc.py <basin_index>")
        print("Example: python distributed_vars_calc.py 1")
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