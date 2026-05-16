import numpy as np
import matplotlib.pyplot as plt
import cdflib.xarray 
import pickle
import os
import glob
import functions.load_routines as load

from functions.spc_current_to_rdf import derive_reduced_vdf

file_path = 'lh_waves_to_doppler_shift.pkl'

with open(file_path, 'rb') as f:
    loaded_dict = pickle.load(f)

loaded_list = loaded_dict['all_lh_to_rh_times']

times_hr = np.unique(loaded_list.astype('datetime64[h]'))
times_day = np.unique(loaded_list.astype('datetime64[D]'))

source_directory = './psp_data/sweap/spc/L2/'

times_day = [np.datetime64('2023-09-26', 'D')]
for date_to_load in times_day: # times_day[0:1]:
    print(date_to_load)
    year, month, day = date_to_load.astype('str').split('-')
    yyyymmdd = f'{year}{month}{day}'
    file_str_start = f'spp_swp_spc_l2i_{yyyymmdd}_v**.cdf'
    search_path = os.path.join(source_directory, str(year), month, file_str_start)
    
    found_file = glob.glob(search_path)

    found_file.sort()
    found_file = found_file[-1]
    
    spc_l2 = load.file_to_xarray(found_file) # cdflib.xarray.cdf_to_xarray(found_file)

    spc_rdf = derive_reduced_vdf(spc_l2)

    # Selected times
    sel_mask = (times_hr.astype('datetime64[D]') == date_to_load)
    times_to_check = times_hr[sel_mask]

    # start_index
    start_index = np.array([np.argmin(np.abs(spc_rdf.epoch.data - time_val)) for time_val in times_to_check])
