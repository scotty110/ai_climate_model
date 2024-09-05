'''
Process Data:
    - Read in the data from the netcdf files
    - Structure the data into a usable format (X,Y pairs)
    - Save data to files
        - Save data to hdf5 file (dictory of X,Y pairs)
        - Save to xarray dataset ()???

--------------------------------

Dir layout:
    - Days (20200101 - 20200110)
        -> 80 regions x 24 time stamps x 8 Files
            -> 30 AVG
            -> 33 AVG
            -> 33 STD
            -> 10 AVG
            -> 10 STD
            -> 408 AVG
            -> 16004 AVG
            -> 16004 STD

--------------------------------

Files Format:
    For use by Daniel Giles at University College London.
    Data produced by Cyril Morcrette at the Met Office (2021).

    There is a data directory for each day (20200101 - 20200110).
    - 80 limited-area models (r01 to r80)
    - 24 time stamps (hourly output)
    - AVG and STD data files for the different data types (see stash codes)

    For each day there are 15360 files.

    In total we have 76800 profiles.

    The filenames are structured as follows:

    20200101T0000Z_r01_km1p5_RA2T_224x224sampling_hence_2x2_time022_STD_16004.nc
                1   2     3    4     5                   6       7   8     9
    1 date/time
    2 which of the 80 limited area models this file is for
    3 the grid-length of the model used to produce the data km1p5 means dx = 1.5 km
    4 the configuration used for the model 
    5 size of region that data is "processed" over. 224x224 means (along with dx=1.5 means we are processing over 336km typical of a climate model grid-box). 
    6 Since each limited area model is run over a large area, despite doing 224x224 averaging, we can do that 2x2 times.
    7 time since the date/time from [1] in steps of 60 minutes. So time000 = 00:00 midnight, and time023 is 23:00
    8 processing that has been done 
    AVG=mean, STD=standard deviation 
    9 see below:
    # List of stash codes to process (Make sure you add any new diagnostics that need processing to all 4 list_* variables.
    # ------------------------------
    # 2d fields (these are 2d fields that do NOT vary with time).
    #  0  30 Land-sea mask
    #  0  33 Orography
    # ------------------------------
    # 4d fields (i.e. 3d fields that vary with time)
    #  0  10 qv          (on theta levels)
    #  0 408 pressure    (on theta levels)
    # 16   4 temperature (on theta levels)
    # ------------------------------

    Each file is then a simple netcdf/xarray file (not loads of metadata).

--------------------------------

Data Layout:
    X:
        Landmas:
            - Land-sea mask (AVG), Orography (AVG, STD)
        Weather:
            - qv (AVG), pressure (AVG), temperature (AVG)
    Y:
        Weather:
            - qv (STD), temperature (STD)
'''

import glob
import re
import h5py

import xarray as xr
import numpy as np
import pandas as pd

from os import listdir
from os.path import isdir, join


'''
Dir Processing
'''
def process_day(path:str) -> list[dict]:
    '''
    Process 1 days worth of data. Read in the data and structure it into X,Y pairs. 
    Inputs:
        - path (str): path to the directory
    Outputs:
        - List of directories (80 regions, 24 timestamps) with the (Landmass,X,Y) pairs
    '''
    # Get day files from
    day = path.split('/')[-1]

    # Get files
    nc_files = glob.glob(join(path, "**/*.nc"), recursive=True)

    # Turn into a list of dictionaries (so then we can load the data into numpy arrays)
    file_list = load_file_list(nc_files, day)
    return file_list


def get_days(path:str) -> list[str]:
    '''
    Get list of days to load (just dirs in the path)
    Inputs:
        - path (str): path to the directory
    Outputs:
        - List of dirs (days) in the path
    '''
    return [name for name in listdir(path) if isdir(join(path, name))]


'''
Data Loading 
'''
def load_nc_data(file:str) -> np.ndarray:
    '''
    Load in the data from a netcdf/xarray file (just value unknown, that is where the data is)
    Inputs:
        - file (str): path to the netcdf file
    Outputs:
        - Numpy array of the data
    '''
    ds = xr.open_dataset(file, engine='netcdf4')
    v = ds['unknown']
    v.load()
    return v.values
    

def load_file_list(nc_files:list[str], day:str) -> list[dict]:
    '''
    Create a pandas dataframe of file names and their attributes.
    Turn that into a list of dictionaries for processing. 
    Inputs:
        - nc_files (list[str]): list of netcdf file paths
        - day (str): day the files are from
    Outputs:
        - List of dictionaries with the keys {day, region, time, files}
    '''
    def extract_parts(filename):
        filename = re.sub(r'\.nc$', '', filename)
        # Split the string on _
        filename_parts = re.split('_', filename)
        filename_parts.remove('hence')
        return filename_parts

    # Create a dataframe with filepath, and attributes to group on
    df = pd.DataFrame(nc_files, columns=['file_path'])
    df[['date_time', 'region', 'grid_length', 'config', 'region_size', 'sampling', 'time', 'processing', 'code']] = df['file_path'].apply(lambda x: pd.Series(extract_parts(x.split('/')[-1])))

    # Group the data by region, time, and code (so we now have groups of files to load)
    simple_df = df[['region', 'time', 'code', 'file_path', 'processing']]
    group_df = simple_df.groupby(['region', 'time',])

    # Create a list of dictionaries of files
    fd_list = []
    for name, group in group_df:
        region, time = name
        file_list = []
        for _, row in group.iterrows():
            file_list.append( row['file_path'] )

        fd_list.append({
            'day': day,
            'region': region,
            'time': time,
            'files': file_list
        })
    return fd_list
    

def load_file_data(file_list:dict) -> dict:
    '''
    Load the data from the netcdf files into numpy arrays.
    Inputs:
        - file_list (dict): dictionary with the keys {day, region, time, files}
    Outputs:
        - Dictionary with the keys {day, region, time, files, arrays}
    '''
    # Get the code of the file (temp, qv, pressure, etc)
    def get_code(s:str) -> str:
        sl = ((s.split('/')[-1]).split('.')[0]).split('_')[-2:] # Should be [processing, code]
        return '_'.join(sl)

    # Loop through the file dictionaries and load the data
    for l in file_list:
        key_file_dict = {}
        for fl in l['files']:
            key_file_dict[get_code(fl)] = fl
        l['files'] = key_file_dict

        # Load the data from the files, and add to the dictionary
        array_dict = {}
        for k in key_file_dict.keys():
            v = key_file_dict[k]
            array_dict[k] = load_nc_data(v)
        l['arrays'] = array_dict
    return file_list

'''
Crop the UM 70 vertical levels to the 32 in CAM
'''
def crop_UM_to_CAM(array: np.ndarray) -> np.ndarray:
    CAM_levels = [3.643466,   7.59482 ,  14.356632,  24.61222 ,  35.92325 ,  43.19375 ,
        51.677499,  61.520498,  73.750958,  87.82123 , 103.317127, 121.547241,
        142.994039, 168.22508 , 197.908087, 232.828619, 273.910817, 322.241902,
        379.100904, 445.992574, 524.687175, 609.778695, 691.38943 , 763.404481,
        820.858369, 859.534767, 887.020249, 912.644547, 936.198398, 957.48548 ,
        976.325407, 992.556095]

    indices = np.zeros((len(CAM_levels), 2, 2), dtype=int)
    # Convert to hPa
    array = array/100.0

    for j in range(len(CAM_levels)):
        indices[j,0,0] = np.argmin(np.abs(CAM_levels - array[j,0,0]))
        indices[j,1,0] = np.argmin(np.abs(CAM_levels - array[j,1,0]))
        indices[j,0,1] = np.argmin(np.abs(CAM_levels - array[j,0,1]))
        indices[j,1,1] = np.argmin(np.abs(CAM_levels - array[j,1,1]))
    return indices.flatten()

'''
Data Processing
'''
def create_pairs(d:dict) -> dict:
    '''
    Create a tuple of ( (lm_avg, Oro_avg, Oro_std), (T_avg, qv_avg, pressure_avg), (T_std, qv_std))
    Inputs:
        - d (dict): dictionary with the keys {day, region, time, files, arrays}
    Outputs:
        - Dictionary with the keys {day, region, time, landmass, x, y}
    '''
    return_dict = {
        'day': d['day'],
        'region': d['region'],
        'time': d['time'],
    }
    d = d['arrays']

    # Get lm, oro
    lm_avg = d['AVG_00030']
    oro_avg = d['AVG_00033']
    oro_std = d['STD_00033']
    return_dict['landmass'] = np.array([lm_avg, oro_avg, oro_std]) # Will need to make sure in same dim

    # Get t_avg, qv_avg, p_avg
    vertical_indices = crop_UM_to_CAM(d['AVG_00408'])

    p_avg = d['AVG_00408'].flatten()[vertical_indices]
    t_avg = d['AVG_16004'].flatten()[vertical_indices]
    qv_avg = d['AVG_00010'].flatten()[vertical_indices]

    p_avg = p_avg.reshape((32, 2, 2))
    t_avg = t_avg.reshape((32, 2, 2))
    qv_avg = qv_avg.reshape((32, 2, 2))
    return_dict['x'] = np.array([t_avg, qv_avg, p_avg])

    # Get t_std, qv_std
    t_std = d['STD_16004'].flatten()[vertical_indices]
    qv_std = d['STD_00010'].flatten()[vertical_indices]

    t_std = t_std.reshape((32, 2, 2))
    qv_std = qv_std.reshape((32, 2, 2))

    return_dict['y'] = np.array([t_std, qv_std])

    return return_dict


def process_file(file_list:list[dict]) -> list[dict]:
    '''
    Process the file list into X,Y pairs
    Inputs:
        - file_list (list[dict]): list of dictionaries with the keys {day, region, time, files, arrays}
    Outputs:
        - List of dictionaries with the keys {day, region, time, landmass, x, y}
    '''
    return [create_pairs(d) for d in file_list]


'''
Save data
'''
def save_h5(data:list[dict], filename:str):
    '''
    Saves a list of dictionaries to an HDF5 file, with dictionaries converted to groups.
    Inputs:
        - data (list[dict]): list of dictionaries to save with keys {day, region, time, landmass, x, y}
        - filename (str): path to the file to save
    Outputs:
        - None, data saved to disk
    ''' 
    with h5py.File(filename, 'w') as f:
        for i, entry in enumerate(data):
            group = f.create_group(f'entry_{i}')
            for key, value in entry.items():
                if isinstance(value, np.ndarray):
                    group.create_dataset(key, data=value)
                else:
                    # Store non-array data as attributes
                    # Ensure that the value is converted to a string, as HDF5 attributes
                    # are more versatile with string data types.
                    group.attrs[key] = str(value)
    return


if __name__ == '__main__':
    file_dir = '/home/squirt/Documents/data/weather_data'

    # Load files into data list of lists 
    days = get_days(file_dir)
    days_data = []
    for day in days:
        day_path = join(file_dir, day)
        file_list = process_day(day_path)
        file_list = load_file_data(file_list)
        data_pairs = process_file(file_list) 
        days_data.append(data_pairs)
    
    # Flatten List of Lists
    flat_data = [item for sublist in days_data for item in sublist]

    # Save to hdf5 file
    save_h5(flat_data, join(file_dir,'all_data.h5'))