import gc
import numpy as np
import cdflib.xarray
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import xarray as xr
import astropy.constants as c

from datetime import datetime
from matplotlib.colors import LogNorm

import functions.load_routines as load
from functions.spc_spike_removal import despike_dataset
from functions.spc_current_to_rdf import derive_reduced_vdf

def drop_spc_repeats(ds):
    """
    Identifies consecutive repeated mv_lo measurements and sets all but 
    the final measurement to NaN across all variables dimensioned by mv_index.
    """
    ds_clean = ds.copy()
    
    # 1. Identify duplicates
    next_mv_lo = ds_clean['mv_lo'].shift(mv_index=-1)
    is_duplicate = np.isclose(ds_clean['mv_lo'], next_mv_lo, atol=0.1) & ~np.isnan(ds_clean['mv_lo'])
    
    # 2. Dynamically find all variables dependent on 'mv_index'
    # ds_clean.variables includes both data_vars and coords.
    # We exclude 'mv_index' to avoid injecting NaNs into the dimension coordinate.
    vars_to_mask = [
        var_name for var_name, var_data in ds_clean.variables.items() 
        if 'mv_index' in var_data.dims and var_name != 'mv_index'
    ]
    
    # 3. Apply the mask
    for var in vars_to_mask:
        ds_clean[var] = ds_clean[var].where(~is_duplicate)
            
    return ds_clean

def get_pA_flags(init_ds, l1_threshold=10, l2_threshold=100):
    """
    Evaluates whether the peak current on ALL FOUR plates exceeds the given thresholds.
    Returns a dataset with a 'total_max_flag' for each epoch.
    """
    keys = ['a', 'b', 'c', 'd']
    max_flags_list = []

    for key in keys:
        # 1. Find the peak signal across the energy dimension for this specific plate
        plate_max = init_ds[f'{key}_current'].max(dim='mv_index', skipna=True)
        
        # 2. Flag the peak (2 if > l2, 1 if > l1, 0 otherwise)
        # We use xr.where to keep everything safely in xarray format
        plate_flag = xr.where(plate_max > l2_threshold, 2, 
                     xr.where(plate_max > l1_threshold, 1, 0))
        
        max_flags_list.append(plate_flag)

    # 3. Concatenate the flags from all 4 plates into a new "flag" dimension
    combined_flags = xr.concat(max_flags_list, dim="flag")

    # 4. Take the minimum across the flag dimension. 
    # This requires ALL 4 plates to have at least a 1 to return a 1.
    ds_stats = xr.Dataset(coords={"epoch": init_ds.epoch})
    ds_stats['total_max_flag'] = combined_flags.min(dim="flag").astype(int)

    return ds_stats

if __name__ == '__main__':
    file_path = 'selected_dates_updated_v2.pkl'

    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    loaded_list = [date for date_list in loaded_dict.values() for date in date_list]
    cutoff = datetime(2022, 1, 1) 
    loaded_list = [d for d in loaded_list if datetime.strptime(d, "%Y-%m-%d") > cutoff]

    # select the data variables. 
    data_vars = {
            'spc' : ['l2i', 'l3i'],
            'fields' : 'mag_SC_4_Sa_per_Cyc',
        }

    fit_seconds = 4 # seconds

    updated_flags = {}

    for date in loaded_list:
        print(f"Processing {date}...")
        trange = [f'{date}T00:00:00', f'{date}T23:59:59']

        try:
            # Load data for a single day
            file_dict = load.load_local_data_file(trange, data_vars, creds=None)
            spc_l2 = load.file_to_xarray(file_dict['spc']['l2i']).rename({'mv_lo_dim' : 'mv_index'})
            # Select only the peak tracks
            peak_track_flag = np.where(spc_l2.MODE_FLAG.data == 0)[0]
            spc_l2 = spc_l2.isel(epoch=peak_track_flag)
            spc_l2 = despike_dataset(spc_l2, threshold=4.0, spikemin=4.0)

            spc_rdf = derive_reduced_vdf(spc_l2)
            spc_rdf = drop_spc_repeats(spc_rdf)

            tstart = trange[0]
            tend   = trange[-1]

            fit_steps = np.timedelta64(np.datetime64(tend, 'ns') - np.datetime64(tstart))/np.timedelta64(fit_seconds, 's')
            fit_time  = np.datetime64(tstart, 'ns') + np.arange(fit_steps+1)*np.timedelta64(fit_seconds, 's')

            windows = np.squeeze(np.dstack([fit_time[0:-1:1], fit_time[1::1]]))            

            init_ds = spc_rdf.copy()
            subset = init_ds[["a_current", "b_current", "c_current", "d_current"]]
            binned_ds = subset.groupby_bins('epoch', fit_time).mean(skipna=True)

            midpoints = binned_ds.epoch_bins.to_index().mid
            binned_ds = binned_ds.assign_coords(epoch=('epoch_bins', midpoints))
            binned_ds = binned_ds.swap_dims({"epoch_bins":"epoch"}).drop_vars("epoch_bins")

            pa_flag = get_pA_flags(binned_ds).rename({'epoch': 'time'})

            updated_flags[f'{date}'] = pa_flag

        except Exception as e:
            print(f"  -> Failed to process {date}: {e}")
            
        finally:
            # 6. EXPLICIT MEMORY MANAGEMENT
            # Delete large objects manually and force the garbage collector
            for var_name in ['spc_l2', 'spc_l3', 'fields_raw', 'fields', 'fields_interp', 'b_vec', 'master_mask']:
                if var_name in locals():
                    del locals()[var_name]
            gc.collect()

    # Save the dictionary of good intervals to a new pickle file
    with open('updated_flags_intervals_v2.pkl', 'wb') as f:
        pickle.dump(updated_flags, f)
        
    print(f"\nFinished! Found valid intervals across {len(updated_flags)} days.")