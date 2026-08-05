import numpy as np
import cdflib.xarray
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

def load_peak_tracks_l2_data(date):
    year, month, day = date.split('-')
    file = f'/home/michael/Research/PSP_Temp_Anisotropy/psp_data/sweap/spc/l2/l2i/{year}/psp_swp_spc_l2i_{year}{month}{day}_v02.cdf'
    spc_l2 = cdflib.xarray.cdf_to_xarray(file, fillval_to_nan=True)
    spc_l2 = spc_l2.rename({'Epoch' : 'epoch'})
    # Get the peak track values only
    peak_track_flag = np.where(spc_l2['MODE_FLAG'].data == 0)[0]

    # Apply the flag
    spc_l2_data = spc_l2.isel(epoch=peak_track_flag)
    
    # Get the length of non-NaN data. If less than 30 not a good peak track!
    len_of_data = np.isfinite(spc_l2_data['a_current'].data).sum(axis=1)
    mask = (len_of_data < 20)

    # Select the high quality peak track data.
    spc_l2_data = spc_l2_data.isel(epoch=~mask)

    return spc_l2_data

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

def despike_spc_hampel(ds, window=5, sigma_threshold=3.0, noise_floor=1e-13):
    """
    Despikes using a robust Hampel Filter (Median Absolute Deviation) 
    and guarantees memory reassignment.
    """
    ds_clean = ds.copy()
    cups = ['a', 'b', 'c', 'd']
    N_epochs = ds.sizes['epoch']
    
    V = ds_clean['mv_lo'].values
    unique_voltages = np.unique(np.round(V[~np.isnan(V)], decimals=2))
    
    for v in unique_voltages:
        row_idx, col_idx = np.nonzero(np.isclose(V, v, atol=0.1))
        if len(row_idx) == 0: continue
            
        for cup in cups:
            # 1. Forcefully copy the array to prevent silent memory issues
            var_name = f'{cup}_current'
            C = ds_clean[var_name].values.copy() 
            
            # 2. Extract 1D array
            ts_1d = np.full(N_epochs, np.nan)
            ts_1d[row_idx] = C[row_idx, col_idx]
            
            # 3. HAMPEL FILTER: Robust Rolling Stats
            ts_series = pd.Series(ts_1d)
            rolling_med = ts_series.rolling(window=window, center=True, min_periods=1).median()
            
            # Calculate Median Absolute Deviation (MAD)
            abs_dev = (ts_series - rolling_med).abs()
            rolling_mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()
            
            # Convert MAD to equivalent Standard Deviation
            rolling_std_approx = rolling_mad * 1.4826
            
            # 4. Spike Identification (incorporating a minimum noise floor)
            threshold = sigma_threshold * np.maximum(rolling_std_approx.values, noise_floor)
            is_spike_1d = np.abs(ts_1d - rolling_med.values) > threshold
            
            # 5. Overwrite and reassign explicitly
            spike_epochs = row_idx[is_spike_1d[row_idx]]
            spike_cols = col_idx[is_spike_1d[row_idx]]
            
            C[spike_epochs, spike_cols] = rolling_med.values[spike_epochs]
            ds_clean[var_name].values = C  # Force the update back into xarray
            
            # 6. Apply same logic to variance arrays
            var_delta = f'{cup}_current_DELTA_VAR'
            if var_delta in ds_clean:
                V_var = ds_clean[var_delta].values.copy()
                var_1d = np.full(N_epochs, np.nan)
                var_1d[row_idx] = V_var[row_idx, col_idx]
                
                var_series = pd.Series(var_1d)
                rolling_med_var = var_series.rolling(window=window, center=True, min_periods=1).median()
                
                V_var[spike_epochs, spike_cols] = rolling_med_var.values[spike_epochs]
                ds_clean[var_delta].values = V_var

    # 7. Re-derive fluxes (now actually using the despiked data)
    total_current = sum([ds_clean[f'{cup}_current'] for cup in cups])
    ds_clean['diff_charge_flux'] = total_current / ds_clean['eff_area']
    
    total_var = sum([ds_clean[f'{cup}_current_DELTA_VAR'] for cup in cups if f'{cup}_current_DELTA_VAR' in ds_clean])
    ds_clean['diff_charge_flux_DELTA_VAR'] = total_var / (ds_clean['eff_area'])
        
    return ds_clean


def despike_spc_idl_style(ds, threshold=4.0, spikemin=4.0, by_step=True, by_spectrum=True, test_mode=False):
    """
    Despikes SPC data mirroring the logic of spp_swp_spc_saltandpepper_filter.pro.
    
    Parameters:
    - ds: xarray Dataset
    - threshold: Relative multiplier threshold for a spike
    - spikemin: Absolute minimum difference to be considered a spike
    - by_step: Boolean to apply filtering across energy steps
    - by_spectrum: Boolean to apply filtering across time/spectra
    - test_mode: Boolean. If True, attaches a dictionary of exact changed 
                 indices and values to ds_clean.attrs['spike_details'].
    """
    ds_clean = ds.copy()
    cups = ['a', 'b', 'c', 'd']
    
    # Initialize dictionary to hold debugging data if test_mode is active
    if test_mode:
        spike_details = {}
    
    # Pre-calculate the steady voltage window flag (samewin) for time-based filtering
    if by_spectrum:
        mv_lo = ds_clean['mv_lo']
        steady_lo = (abs(mv_lo - mv_lo.shift(epoch=1)) < 1) & (abs(mv_lo - mv_lo.shift(epoch=-1)) < 1)
        
        if 'mv_hi' in ds_clean:
            mv_hi = ds_clean['mv_hi']
            steady_hi = (abs(mv_hi - mv_hi.shift(epoch=1)) < 1) & (abs(mv_hi - mv_hi.shift(epoch=-1)) < 1)
            samewin = steady_lo & steady_hi
        else:
            samewin = steady_lo

    for cup in cups:
        var_name = f'{cup}_current'
        if var_name not in ds_clean:
            continue
            
        data = ds_clean[var_name]
        
        # Initialize sub-masks for this specific cup
        is_spike_step = xr.zeros_like(data, dtype=bool)
        is_spike_time = xr.zeros_like(data, dtype=bool)
        
        # ---------------------------------------------------------
        # Filter by Step (across mv_index / energy)
        # ---------------------------------------------------------
        if by_step:
            min_val = abs(data).min().values 
            median_step = data.rolling(mv_index=3, center=True, min_periods=1).median()
            diff_step = data - median_step
            
            is_spike_step = (
                (diff_step > (threshold * median_step)) & 
                (median_step > 0) & 
                (diff_step > spikemin)
            ) | (data == min_val)
            
            data = xr.where(is_spike_step, median_step, data)
            
        # ---------------------------------------------------------
        # Filter by Spectrum (across epoch / time)
        # ---------------------------------------------------------
        if by_spectrum:
            median_time = data.rolling(epoch=3, center=True, min_periods=1).median()
            diff_time = data - median_time
            
            is_spike_time = (
                (diff_time > (threshold * median_time)) & 
                (median_time > 0) & 
                (diff_time > spikemin) & 
                samewin
            )
            
            data = xr.where(is_spike_time, median_time, data)
            
        # ---------------------------------------------------------
        # Store Testing / Debugging Details
        # ---------------------------------------------------------
        if test_mode:
            # Combine both masks to find every location a spike was removed for THIS cup
            cup_mask = is_spike_step | is_spike_time
            
            # Extract 2D integer indices where spikes occurred 
            # (Assuming standard dimensions of ('epoch', 'mv_index'))
            idx_epoch, idx_mv = np.where(cup_mask.values)
            
            # Store the indices and their corresponding physical values
            spike_details[cup] = {
                'epoch_indices': idx_epoch,
                'mv_indices': idx_mv,
                'epoch_values': ds_clean['epoch'].values[idx_epoch],
                'mv_index_values': ds_clean['mv_index'].values[idx_mv]
            }

        # Push cleaned data back to dataset
        ds_clean[var_name] = data
        
        # ---------------------------------------------------------
        # Apply the mask to DELTA_VAR arrays
        # ---------------------------------------------------------
        var_delta = f'{cup}_current_DELTA_VAR'
        if var_delta in ds_clean:
            data_var = ds_clean[var_delta]
            
            if by_step:
                median_var_step = data_var.rolling(mv_index=3, center=True, min_periods=1).median()
                data_var = xr.where(is_spike_step, median_var_step, data_var)
            
            if by_spectrum:
                median_var_time = data_var.rolling(epoch=3, center=True, min_periods=1).median()
                data_var = xr.where(is_spike_time, median_var_time, data_var)
                
            ds_clean[var_delta] = data_var

    # Attach the testing details to the dataset attributes
    if test_mode:
        ds_clean.attrs['spike_details'] = spike_details

    # ---------------------------------------------------------
    # Re-derive fluxes
    # ---------------------------------------------------------
    total_current = sum([ds_clean[f'{cup}_current'] for cup in cups if f'{cup}_current' in ds_clean])
    if 'eff_area' in ds_clean:
        ds_clean['diff_charge_flux'] = total_current / ds_clean['eff_area']
    
    total_var = sum([ds_clean[f'{cup}_current_DELTA_VAR'] for cup in cups if f'{cup}_current_DELTA_VAR' in ds_clean])
    if 'eff_area' in ds_clean:
        ds_clean['diff_charge_flux_DELTA_VAR'] = total_var / ds_clean['eff_area']
        
    return ds_clean

def add_effective_area(ds):
    """Helper function to add in effective area from SPC measurements"""
    itot = ds.a_current + ds.b_current + ds.c_current + ds.d_current
    eff_area = itot.data/ds.diff_charge_flux_density.data

    ds['eff_area'] = (['epoch', 'mv_index'], eff_area)

    return ds

def despike_dataset(ds, threshold=4.0, spikemin=4.0, test_mode=False):
    if 'mv_lo_dim' in ds.dims:
        ds = ds.rename({'mv_lo_dim':'mv_index'})
    
    # Drop repeates
    ds = drop_spc_repeats(ds)
    ds = add_effective_area(ds)

    ds_clean = despike_spc_idl_style(ds, threshold=threshold, spikemin=spikemin, test_mode=test_mode)

    return ds_clean

if __name__ == "__main__":
    date = '2023-09-26'

    ds = load_peak_tracks_l2_data(date)
    # ds = ds.rename({'mv_lo_dim':'mv_index'})
    # ds = drop_spc_repeats(ds)
    # ds = add_effective_area(ds)

    # ds_clean = despike_spc_hampel(ds, window=5, sigma_threshold=3)

    ds_clean = despike_dataset(ds.copy(), test_mode=True)

