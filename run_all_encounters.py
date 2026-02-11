# This function is used to run all the psp encounters calling the calc_spc_temp_aniso function to run
# the code.
import os
import gc
import sys
import json
import glob
import pickle
import cdflib
import pyspedas
import datetime
import traceback
import cdflib.xarray

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u

from tqdm import tqdm
from matplotlib.colors import LogNorm

from datetime import datetime

# Import user defined functions
import functions.data_preprocessing as preprocess
import functions.load_routines as load

from functions.spc_current_to_rdf import derive_reduced_vdf, kp_file_to_derived_reduced_vdf
from support_functions.gen_spc_uncert_map_2d import gen_uncertainty_map3

# TODO : Revisit this function to make sure the calculation is working correctly.
from functions.spc_do_anisotropy_calc import perform_fitting_with_time_windows
from functions.data_preprocessing import xarray_clean_variable as clean

from functions.calc_spc_temp_aniso import process_anisotorpy_spc_event

from functions.misc_functions import load_config

if __name__ == "__main__":
    pyspedas.psp.config.local_data_dir = 'psp_data'

    # 1. Configuration
    config = {
        # 'fit_seconds':4,
        'fit_seconds': None,
        'smooth_seconds': None,
        # 'spc_mode': 'peak_track',
        'spc_mode': 'full_scan',
        'no_weights': None,
        'win_max': False,
        'min_flux': 0.5,
        'avg_bfield': False,
        'data_vars': {
            'spc' : ['l2i', 'l3i'],
            'fields' : 'mag_SC_4_Sa_per_Cyc',
            'spi_sf00' : ['L3']
        },
        'credentials': load_config('config.json'),
        'uncert_map': gen_uncertainty_map3(4, 30),
        'OFFLINE': True,
    }
    
    file_path = 'selected_dates.pkl'

    # Load the list (open file in 'rb' binary read mode)
    with open(file_path, 'rb') as f:
        loaded_list = pickle.load(f)

    # dates_to_process = ['2024-06-23', '2024-06-24', '2024-06-25' , '2024-06-26', '2024-06-27']
    # dates_to_process = ['2024-06-27']
    # dates_to_process = loaded_list

    cutoff = datetime(2022,1,1) 

    dates_to_process = [
        d for d in loaded_list
        if datetime.strptime(d, "%Y-%m-%d") > cutoff
    ]


    # 3. Iterate
    print(f"Queued {len(dates_to_process)} days for processing.")
    
    for date_str in tqdm(dates_to_process):
        process_anisotorpy_spc_event(date_str, config)

    print('All events completed!')
