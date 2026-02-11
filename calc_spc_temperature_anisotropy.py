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
from support_functions.gen_spc_uncert_map_2d import gen_uncertainty_map

# TODO : Revisit this function to make sure the calculation is working correctly.
from functions.spc_do_anisotropy_calc import perform_fitting_with_time_windows
from functions.data_preprocessing import xarray_clean_variable as clean

from functions.misc_functions import load_config, hampel_filter2
from functions.smooth_spc_data import smooth_spc_resample

from functions.spc_calc_anisotropy_v3 import spc_calculate_anisotropy, spc_calculate_anisotropy_groups
from functions.spc_calc_anisotropy_v3 import calculate_support_means_groups, get_support_values

from scipy.optimize import curve_fit

plt.ion()

def spc_select_key_values(spc_l2):
    spc = xr.Dataset(
        data_vars=dict(
            mv_lo = (['time', 'dim'], spc_l2.mv_lo.data),
            mv_hi = (['time', 'dim'], spc_l2.mv_hi.data),
            a_current = (['time', 'dim'], spc_l2.a_current.data),
            b_current = (['time', 'dim'], spc_l2.b_current.data),
            c_current = (['time', 'dim'], spc_l2.c_current.data),
            d_current = (['time', 'dim'], spc_l2.d_current.data),
            diff_charge_flux_density = (['time', 'dim'], spc_l2.diff_charge_flux_density.data)
        ),
        coords=dict(
            epoch = spc_l2.epoch.data,
            dim = np.arange(spc_l2.a_current.data.shape[1])
        ),
        attrs=dict(Note = "Selected Data Quantities")
    )

    return spc

def median_filter_current(spc_l2):
    current_stack = np.stack([spc_l2.a_current.data, spc_l2.b_current.data, spc_l2.c_current.data, spc_l2.d_current.data])

    # set all fillvals to NaNs
    current_stack[current_stack < 0] = np.nan

    median_current = np.median(current_stack, axis=0)

    for i in range(4):
        current_stack[i][current_stack[i] > 3*median_current] = median_current[current_stack[i] > 3*median_current]

    vals = ['a', 'b', 'c', 'd']
    for i, val in enumerate(vals):
        spc_l2[f'{val}_current'].data = current_stack[i]

    return(spc_l2)

def initialize_fitting_data(trange, data_vars, smooth_seconds=None, mode='all', credentials=None, SAVE=True, FPATH='./input_data/', KP_FILE=None, OFFLINE=None):
    btag = data_vars['fields'].removeprefix('mag_SC_')


    if KP_FILE:
        print('Loading in downsampled data from Kristoff.')
        spc_l2_data = cdflib.xarray.cdf_to_xarray(KP_FILE, to_datetime=True)
        load_l2 = False
    elif smooth_seconds:
        # Now we need to preform the data smoothing.
        # First let us check if the smooth_second data exists.
        fname0 = f'SPC_downsampled_spectra_{trange[0][:10]}_'
        time_tag = int(smooth_seconds) 
        time_unit = 's'
        if smooth_seconds % 1 != 0:
            print('Adjusting smooth_seconds tag to millisecond resolution.')
            #Convert timetag to ms.
            time_tag = int(smooth_seconds * 1000)
            time_unit = 'ms'

        if mode.lower() != 'peak_track':
            time_unit = time_unit+f'_{mode.lower()}'
        else:
            time_unit = time_unit
        fname = fname0+f'{time_tag}{time_unit}.cdf'
        # print(f'{FPATH}{fname}')
        if os.path.exists(FPATH+fname):
            print(f'{time_tag}{time_unit} file exists. Loading in downsampled dataset.')
            spc_red = load.file_to_xarray(f'{FPATH}{fname}')
            spc_red = spc_red.where(spc_red != -1e31, np.nan)

            # drop the l2i data var as we are using the reduced data array
            data_vars['spc'] = 'l3i'
            load_l2 = False

            spc_l2_data = spc_red
        else:
            load_l2 = True
    else:
        load_l2 = True

    if OFFLINE:
        file_dict = load.load_local_data_file(trange, datasets=data_vars, creds=credentials)
    else:
        # Can group into a single function. 
        file_dict = load.load_data(mission='psp', datasets=data_vars, trange=trange, creds=credentials)

    print(f'\n {file_dict['spc']['l3i']} \n')
    # Load in the l3 data set. This is fully a support function.
    spc_l3 = load.file_to_xarray(file_dict['spc']['l3i'])

    # Load in the field data set.
    fields = load.file_to_xarray(file_dict['fields'][data_vars['fields']])

    if load_l2: 
        print('Loading in the l2 data')
        spc_l2 = load.file_to_xarray(file_dict['spc']['l2i']).rename({'mv_lo_dim' : 'mv_index'})
        spc_l2 = median_filter_current(spc_l2)

        if  mode.lower() == 'full_scan':
            full_scan_flag = np.where(spc_l2.MODE_FLAG.data == 1)[0]
            spc_l2_data = spc_l2.isel(epoch=full_scan_flag)
        elif mode.lower() == 'peak_track':
            peak_track_flag = np.where(spc_l2.MODE_FLAG.data == 0)[0]
            spc_l2_data = spc_l2.isel(epoch=peak_track_flag)
        else:
            spc_l2_data = spc_l2

    if smooth_seconds and load_l2:
        spc_red = smooth_spc_resample(spc_l2, smooth_seconds)
        spc_red = spc_red.drop_attrs()
        if SAVE:
            print('Saving downsampled data to xarray for future use.')
            cdflib.xarray.xarray_to_cdf(spc_red, f'./input_data/SPC_downsampled_spectra_{trange[0][:10]}_{time_tag}{time_unit}')
    
        spc_l2_data = spc_red

    if KP_FILE:
        spc_rdf = kp_file_to_derived_reduced_vdf(spc_l2_data)
    else:
        spc_rdf = derive_reduced_vdf(spc_l2_data)   # derive the rdf data.
    
    # Now that the SPC has been delt with. let us fix the bfield
    if 'psp_fld_l2_mag_SC_4_Sa_per_Cyc' in fields.data_vars:
        fields = clean(fields['psp_fld_l2_mag_SC_4_Sa_per_Cyc'])
    else:    
        fields = clean(fields['psp_fld_l2_mag_SC_1min'])
    
    # hampel filter the fields data
    fields.data = hampel_filter2(fields.data, 5)[0]

    # Interpolate the fields data onto the same time resolution.
    flds_inter  = fields.interp(epoch=spc_rdf.epoch)
    spc_rdf['b_vec'] = flds_inter

    # Let us attach the l3 supporting data for density, velocity, and temperature.
    support_vars = ['np_fit', 'wp_fit', 'vp_fit_SC', 'np_moment', 'wp_moment', 'vp_moment_SC', 'general_flag']

    for var in support_vars:
        spc_rdf[var] = spc_l3[var]
    # Add in current flags
    
    return(spc_rdf)

def get_pA_flags(init_ds):
    keys = ['a', 'b', 'c', 'd']
    
    max_dict = {}
    med_dict = {}

    for key in keys:
        max_dict[f'{key}_max'] = np.nanmax(init_ds[f'{key}_current'], axis=1)
        med_dict[f'{key}_med'] = np.nanmedian(init_ds[f'{key}_current'], axis=1)

    max_flag = {}
    med_flag = {}
    for key in keys:
        max_flag[f'{key}_max_flag'] = np.where(max_dict[f'{key}_max'] > 100, 2, np.where(max_dict[f'{key}_max'] > 10, 1, 0))
        med_flag[f'{key}_med_flag'] = np.where(med_dict[f'{key}_med'] > 100, 2, np.where(med_dict[f'{key}_med'] > 10, 1, 0))

    data_dict = {**max_dict, **med_dict, **max_flag, **med_flag}

    ds_stats = xr.Dataset(
        data_vars={k : (("epoch",), v) for k, v in data_dict.items()},
        coords={"epoch": init_ds.epoch}
    )

    flags_max = xr.concat([ds_stats['a_max_flag'], ds_stats['b_max_flag'], ds_stats['c_max_flag'], ds_stats['d_max_flag']], dim="flag")
    flags_med = xr.concat([ds_stats['a_med_flag'], ds_stats['b_med_flag'], ds_stats['c_med_flag'], ds_stats['d_med_flag']], dim="flag")

    ds_stats['total_max_flag'] = (flags_max > 0).all(dim="flag").astype(int)
    ds_stats['total_med_flag'] = (flags_med > 0).all(dim="flag").astype(int)

    return(ds_stats)

def time_index(i):
    if i == 0:
        print("Encounter 17")
        tstart = '2023-09-26T00:00:00' 
        tend   = '2023-09-26T12:00:00'
    if i == 1:
        print("Encounter 18")
        tstart = '2023-12-27T00:00:00' 
        tend   = '2023-12-27T07:59:59'
    if i == 2:
        print("Encounter 19")
        tstart = '2024-03-28T00:00:00' 
        tend   = '2024-03-28T04:59:59'
    if i == 3:
        print("Encounter 20")
        tstart = '2024-06-27T00:00:00' 
        tend   = '2024-06-27T23:59:59'
    if i == 4:
        print("Encounter 21")
        tstart = '2024-09-27T00:00:00' 
        tend   = '2024-09-27T23:59:59'
    if i == 5:
        print("Encounter 22")
        tstart = '2024-12-22T00:00:00' 
        tend   = '2024-12-22T23:59:59'

    return(tstart, tend)

# This is where the core processing function will occur. 
# NOTE: This runs an entire day at a time.
def process_anisotorpy_spc_event(date_str, config_dict, subrange = None):
    '''
    Docstring for process_anisotorpy_spc_event
    
    :param date_str: Description
    :param config_dict: Description
    :param subrage:
    '''
    if subrange is not None:
        tstart = subrange[0]
        tend   = subrange[1]
        range_desc = f"Subrange ({tstart} to {tend})"
        is_subrange = True
    else:
        tstart = f"{date_str}T00:00:00"
        tend   = f"{date_str}T23:59:59"
        range_desc = "Full Day"
        is_subrange = False
    
    # Create full data timestamps
    # tstart = f"{date_str}T00:00:00"
    # tend   = f"{date_str}T23:59:59"

    print(f"\n{'='*50}")
    print(f"PROCESSING: {date_str}")
    print(f"WINDOW:     {range_desc}")
    print(f"{'='*50}")

    try:
        # Unpack config
        fit_seconds = config_dict['fit_seconds']
        smooth_seconds = config_dict['smooth_seconds']
        spc_mode = config_dict['spc_mode']
        no_weights = config_dict['no_weights']
        win_max = config_dict['win_max']
        min_flux = config_dict['min_flux']
        avg_bfield = config_dict['avg_bfield']
        data_vars = config_dict['data_vars']
        credentials = config_dict['credentials']
        uncert_map = config_dict['uncert_map']

        OFFLINE = config_dict['OFFLINE']
        
        trange = [tstart, tend]

        # Initialize Data 
        # (This likely loads the full daily CDF anyway, matching your requirements)
        init_ds = initialize_fitting_data(trange, data_vars, smooth_seconds=smooth_seconds, mode=spc_mode, credentials=credentials, OFFLINE=OFFLINE)

        # This ensures that if the loader returned a full day, we only process the subrange.
        print(f"Slicing data to range: {tstart} -- {tend}")
        init_ds = init_ds.sel(epoch=slice(tstart, tend))

        # Check if data exists in the window
        if init_ds.epoch.size == 0:
            print(f"No data found in range {tstart} to {tend}. Skipping.")
            return

        # Get pico-Amp flags
        pa_flag = get_pA_flags(init_ds).rename({'epoch':'time'})
        
        # We need to construct the fit windows based on the actual loaded data range
        # OR strictly enforce the day boundaries. 
        # Using the actual start/end from trange ensures we cover the requested day.
        if fit_seconds != None:
            fit_steps = np.timedelta64(np.datetime64(tend, 'ns') - np.datetime64(tstart))/np.timedelta64(fit_seconds, 's')
            fit_time  = np.datetime64(tstart, 'ns') + np.arange(fit_steps+1)*np.timedelta64(fit_seconds, 's')

            windows = np.squeeze(np.dstack([fit_time[0:-1:1], fit_time[1::1]]))

            # Optimization: drop unused vars
            subset = init_ds.drop_vars([v for v in init_ds.data_vars if v not in ["a_current", "b_current", "c_current", "d_current"]])
            ds_groupby = subset.groupby(['epoch', 'mv_lo']).mean(skipna=True)
            binned_ds = ds_groupby.groupby_bins('epoch', fit_time).mean(skipna=True)

            midpoints = binned_ds.epoch_bins.to_index().mid
            binned_ds = binned_ds.assign_coords(epoch=('epoch_bins', midpoints))
            binned_ds = binned_ds.swap_dims({"epoch_bins":"epoch"}).drop_vars("epoch_bins")

            pa_flag = get_pA_flags(binned_ds).rename({'epoch': 'time'})

            if no_weights:
                spc_anisotropy = spc_calculate_anisotropy_groups(init_ds, uncert_map, threshold=min_flux, NO_WEIGHTS=no_weights, groups=windows, WIN_MAX=win_max, USE_AVG_BFIELD=avg_bfield)    
            else:
                spc_anisotropy = spc_calculate_anisotropy_groups(init_ds, uncert_map, threshold=min_flux, groups=windows, WIN_MAX=win_max, USE_AVG_BFIELD=avg_bfield)

            support_spc = calculate_support_means_groups(init_ds, groups=windows).rename({'epoch' : 'time'})
        else:
            if no_weights:
                spc_anisotropy = spc_calculate_anisotropy(init_ds, uncert_map, NO_WEIGHTS=True, threshold=min_flux)
            else:
                spc_anisotropy = spc_calculate_anisotropy(init_ds, uncert_map, threshold=min_flux)

            support_spc = get_support_values(init_ds).rename({'epoch' : 'time'})

        spc_anisotropy = xr.merge([spc_anisotropy, pa_flag, support_spc])

        # Build Filename
        tags = []
        if smooth_seconds:
            if isinstance(smooth_seconds, int):
                tags.append(f"{int(smooth_seconds)}s")
            elif smooth_seconds % 1 != 0:
                print("Adjusting smooth_seconds tag to millisecond resolution.")
                smooth_seconds_ms = int(smooth_seconds * 1000)
                tags.append(f"{smooth_seconds_ms}ms")

        if fit_seconds:
            tags.append(f"averaged_{fit_seconds}s")

        if no_weights:
            tags.append('nw')

        if win_max:
            tags.append('winmax')

        tag_str = "_" + "_".join(tags) if tags else ""

        # Using the DATE for filename
        output_dir = "./anisotropy_files"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if avg_bfield:
            filename = f"{output_dir}/SPC_Anisotropy_{date_str}_{spc_mode}_bavg_{tag_str}"
        else:
            filename = f"{output_dir}/SPC_Anisotropy_{date_str}_{spc_mode}{tag_str}"

        # 5. Save
        print(f"Saving to {filename}...")
        cdflib.xarray.xarray_to_cdf(spc_anisotropy, filename)
        
    except Exception as e:
        print(f"!!! Error processing day {date_str} !!!")
        print(e)
        traceback.print_exc()
        
    finally:
        # 6. Cleanup Memory
        if 'init_ds' in locals(): del init_ds
        if 'spc_anisotropy' in locals(): del spc_anisotropy
        if 'subset' in locals(): del subset
        
        # Force garbage collection
        gc.collect()

if __name__ == "__main__":
    pyspedas.psp.config.local_data_dir = 'psp_data'
    pyspedas.psp.config.CONFIG['sweap_remote_data_dir'] = 'http://131.142.18.203/data/sci/'

    # 1. Configuration
    config = {
        'fit_seconds':4,
        # 'fit_seconds': None,
        'smooth_seconds': None,
        'spc_mode': 'peak_track',
        # 'spc_mode': 'full_scan',
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
        'uncert_map': gen_uncertainty_map(4, 30),
        'OFFLINE':False, 
    }
    
    file_path = 'selected_dates.pkl'

    # Load the list (open file in 'rb' binary read mode)
    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    loaded_list = [date for date_list in loaded_dict.values() for date in date_list]

    # dates_to_process = ['2024-06-23', '2024-06-24', '2024-06-25' , '2024-06-26', '2024-06-27']
    dates_to_process = ['2023-12-27']
    # dates_to_process = loaded_list

    ds = process_anisotorpy_spc_event(dates_to_process[0], config, subrange=['2023-12-27T04:30:00', '2023-12-27T04:45:00'])
    '''
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
    '''
