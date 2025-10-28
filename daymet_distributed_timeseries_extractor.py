import pandas as pd
import numpy as np
import datetime
import os
import xarray as xr
import sys

# ----------------------------------------------------------------------------------
# Need changing
csv_folder = '/icebox/data/shares/mh2/mosavat/Distributed/test_targets'
output_directory = '/icebox/data/shares/mh2/mosavat/Distributed/test_inputs'
basin_txt_file = '/icebox/data/shares/mh2/mosavat/Distributed/test_list.txt'
# ----------------------------------------------------------------------------------
cells_folder = '/icebox/data/shares/mh2/mosavat/Distributed/Basin_Cells/files'
daymet_folder = '/icebox/data/shares/mh1/mosavat/Runoff_Generation_Project/Daymet_Data_for_CONUS'

with open(basin_txt_file, 'r') as f:
    basin_list = [line.strip() for line in f]

variables = ['prcp', 'tmin', 'tmax', 'srad', 'vp']

class InputExtractor:
    def __init__(self, basin, csv_folder):
        self.basin = basin
        self.csv_folder = csv_folder
        self.variables = variables
        self.cells_folder = cells_folder
        self.daymet_folder = daymet_folder
        self.output_dir = output_directory

    def _csv_loader(self, basin):
        df = pd.read_csv(f'{self.csv_folder}/{basin}.csv')
        return df

    def start_end_date(self, basin):
        df = self._csv_loader(basin)
        start_date = df['datetime'].iloc[0]
        end_date = df['datetime'].iloc[-1]
        return(start_date, end_date)

    def _npy_loader(self, npy_file):
        file = np.load(npy_file)
        return(file)

    def _nc_loader(self, filepath):
        data = xr.open_dataset(filepath)
        return(data)

    def output_processor(self, basin):
        basin_cells = self._npy_loader(f'{self.cells_folder}/{basin}.npy')
        start_date, end_date = self.start_end_date(basin)
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        start_date_index = pd.to_datetime(start_date).timetuple().tm_yday - 1
        no_of_days = (end_year - start_year + 1) * 365
        arr = np.full((61, 61, 5, no_of_days), -999, dtype=np.float32)
        frame_dimension = 61
        for i in range(len(self.variables)):
            variable = self.variables[i]
            for year in range(start_year, end_year+1):
                nc_file = self._nc_loader(f'{daymet_folder}/daymet_v4_daily_na_{variable}_{year}.nc')
                yr_index = (year - start_year)*365
                arr[:, :, i, yr_index:yr_index+365] = nc_file[variable][:, basin_cells[1,0]:basin_cells[1,0]+frame_dimension, basin_cells[0,0]:basin_cells[0,0]+frame_dimension].transpose('y', 'x', 'time')
        clipped_arr = arr[:, :, :, start_date_index:start_date_index+3650]
        return(clipped_arr)
        
    def npy_save(self, basin):
        """
        Processes the data and saves the final .npy file, 
        but only if the target file does not already exist.
        """
        # 1. Define the expected output file path
        output_filepath = os.path.join(self.output_dir, f'{basin}.npy')
        
        # 2. CHECK: If file exists, skip processing
        if os.path.exists(output_filepath):
            print(f"Skipping basin {basin}: File already exists at {output_filepath}")
            return # Exit the function early
        
        # If the file does not exist, proceed with processing
        print(f"Processing basin: {basin}...")
        
        try:
            # 3. Process the data
            arr = self.output_processor(basin)
        except Exception as e:
            # Handle cases where processing fails (e.g., missing CSV/NPY)
            print(f"Error processing basin {basin}: {e}")
            return

        # 4. Final checks and saving
        if np.isnan(arr).any():
            print(f'Basin {basin} contains NaN values and was NOT saved.')
            with open('/icebox/data/shares/mh2/mosavat/Distributed/distributed_nan.txt', 'a') as f:
                f.write(f'{basin}\n')
        elif arr.shape != (61, 61, 5, 3650):
            print(f'The shape of array for basin {basin} is {arr.shape} (expected (61, 61, 5, 3650)) and was NOT saved.')
        else:
            np.save(output_filepath, arr)
            print(f"Successfully saved file: {output_filepath} with shape {arr.shape}")

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    basin = basin_list[task_id]
    
    extractor = InputExtractor(basin=basin, csv_folder=csv_folder)
    extractor.npy_save(basin)