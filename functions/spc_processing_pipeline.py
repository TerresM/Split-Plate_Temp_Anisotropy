# functions/spc_processing_pipeline.py

import os
import gc
import traceback
import numpy as np
import xarray as xr
import cdflib.xarray

# Import user defined functions (Assuming the execution script is run from the root directory)
import functions.load_routines as load
from functions.spc_current_to_rdf import derive_reduced_vdf, kp_file_to_derived_reduced_vdf
from functions.data_preprocessing import xarray_clean_variable as clean
from functions.misc_functions import hampel_filter2
from functions.smooth_spc_data import smooth_spc_resample

# from functions.spc_calc_anisotropy_v3 import (
#     spc_calculate_anisotropy, 
#     spc_calculate_anisotropy_groups,
#     calculate_support_means_groups, 
#     get_support_values
# )

from functions.spc_calc_anisotropy import calculate_spc_anisotropy

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

    return spc_l2

def initialize_fitting_data(trange, data_vars, smooth_seconds=None, mode='all', credentials=None, SAVE=True, FPATH='./input_data/', KP_FILE=None, OFFLINE=None):
    btag = data_vars['fields'].removeprefix('mag_SC_')

    if KP_FILE:
        print('Loading in downsampled data from Kristoff.')
        spc_l2_data = cdflib.xarray.cdf_to_xarray(KP_FILE, to_datetime=True)
        load_l2 = False
    elif smooth_seconds:
        fname0 = f'SPC_downsampled_spectra_{trange[0][:10]}_'
        time_tag = int(smooth_seconds) 
        time_unit = 's'
        if smooth_seconds % 1 != 0:
            print('Adjusting smooth_seconds tag to millisecond resolution.')
            time_tag = int(smooth_seconds * 1000)
            time_unit = 'ms'

        if mode.lower() != 'peak_track':
            time_unit = time_unit+f'_{mode.lower()}'
            
        fname = fname0+f'{time_tag}{time_unit}.cdf'
        
        if os.path.exists(FPATH+fname):
            print(f'{time_tag}{time_unit} file exists. Loading in downsampled dataset.')
            spc_red = load.file_to_xarray(f'{FPATH}{fname}')
            spc_red = spc_red.where(spc_red != -1e31, np.nan)

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
        file_dict = load.load_data(mission='psp', datasets=data_vars, trange=trange, creds=credentials)

    print(f"\n {file_dict['spc']['l3i']} \n")
    spc_l3 = load.file_to_xarray(file_dict['spc']['l3i'])
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
        spc_rdf = derive_reduced_vdf(spc_l2_data)
    
    if 'psp_fld_l2_mag_SC_4_Sa_per_Cyc' in fields.data_vars:
        fields = clean(fields['psp_fld_l2_mag_SC_4_Sa_per_Cyc'])
    else:    
        fields = clean(fields['psp_fld_l2_mag_SC_1min'])
    
    fields.data = hampel_filter2(fields.data, 5)[0]
    flds_inter  = fields.interp(epoch=spc_rdf.epoch)
    spc_rdf['b_vec'] = flds_inter

    support_vars = ['np_fit', 'wp_fit', 'vp_fit_SC',
                    'np1_fit', 'wp1_fit', 'vp1_fit_SC',
                    'n3_fit', 'w3_fit', 'v3_fit_SC',
                    'na_fit', 'wa_fit', 'va_fit_SC',
                    'np_moment', 'wp_moment', 'vp_moment_SC',
                    'general_flag']

    for var in support_vars:
        spc_rdf[var] = spc_l3[var]

    return spc_rdf

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

    ds_stats['total_max_flag'] = flags_max.min(dim="flag").astype(int)
    ds_stats['total_med_flag'] = flags_med.min(dim="flag").astype(int)

    return ds_stats

def time_index(i):
    encounters = {
        0: ("Encounter 17", '2023-09-26T00:00:00', '2023-09-26T12:00:00'),
        1: ("Encounter 18", '2023-12-27T00:00:00', '2023-12-27T07:59:59'),
        2: ("Encounter 19", '2024-03-28T00:00:00', '2024-03-28T04:59:59'),
        3: ("Encounter 20", '2024-06-27T00:00:00', '2024-06-27T23:59:59'),
        4: ("Encounter 21", '2024-09-27T00:00:00', '2024-09-27T23:59:59'),
        5: ("Encounter 22", '2024-12-22T00:00:00', '2024-12-22T23:59:59'),
    }
    if i in encounters:
        print(encounters[i][0])
        return encounters[i][1], encounters[i][2]
    return None, None

def process_anisotorpy_spc_event(date_str, config_dict, subrange=None):
    if subrange is not None:
        tstart, tend = subrange
        range_desc = f"Subrange ({tstart} to {tend})"
        slice_things = True
    else:
        tstart = f"{date_str}T00:00:00"
        tend   = f"{date_str}T23:59:59"
        range_desc = "Full Day"
        slice_things = False
    
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
        gen_flag   = config_dict['gen_flag']
        OFFLINE = config_dict['OFFLINE']
        
        trange = [tstart, tend]

        init_ds = initialize_fitting_data(trange, data_vars, smooth_seconds=smooth_seconds, mode=spc_mode, credentials=credentials, OFFLINE=OFFLINE)

        if gen_flag:
            flag = init_ds.general_flag.data
            init_ds = init_ds.isel(epoch = np.where(flag == 0)[0])
            print(f"Dropping data with general_flag != 0! (Total Dropped)/Total = {len(flag) - np.sum(flag == 0)}/{len(flag)}")

        if slice_things:
            print(f"Slicing data to range: {tstart} -- {tend}")
            init_ds = init_ds.sel(epoch=slice(tstart, tend))
            
        if init_ds.epoch.size == 0:
            print(f"No data found in range {tstart} to {tend}. Skipping.")
            return
        
        pa_flag = get_pA_flags(init_ds).rename({'epoch':'time'})
        """
        if fit_seconds is not None:
            fit_steps = np.timedelta64(np.datetime64(tend, 'ns') - np.datetime64(tstart))/np.timedelta64(fit_seconds, 's')
            fit_time  = np.datetime64(tstart, 'ns') + np.arange(fit_steps+1)*np.timedelta64(fit_seconds, 's')
            windows = np.squeeze(np.dstack([fit_time[0:-1:1], fit_time[1::1]]))

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
        """
        # Safely extract the plotting flag (defaults to False if not in config)
        plot_diagnostics = config_dict.get('plot_diagnostics', False)

        # Define the support variables we want to carry over to the final dataset
        support_vars = ['np_fit', 'wp_fit', 'vp_fit_SC', 'np1_fit', 'wp1_fit', 'vp1_fit_SC', 
                        'n3_fit', 'w3_fit', 'v3_fit_SC', 'na_fit', 'wa_fit', 'va_fit_SC', 
                        'np_moment', 'wp_moment', 'vp_moment_SC', 'general_flag']

        if fit_seconds is not None:
            fit_steps = np.timedelta64(np.datetime64(tend, 'ns') - np.datetime64(tstart))/np.timedelta64(fit_seconds, 's')
            fit_time  = np.datetime64(tstart, 'ns') + np.arange(fit_steps+1)*np.timedelta64(fit_seconds, 's')
            windows = np.squeeze(np.dstack([fit_time[0:-1:1], fit_time[1::1]]))

            subset = init_ds.drop_vars([v for v in init_ds.data_vars if v not in ["a_current", "b_current", "c_current", "d_current"]])
            ds_groupby = subset.groupby(['epoch', 'mv_lo']).mean(skipna=True)
            binned_ds = ds_groupby.groupby_bins('epoch', fit_time).mean(skipna=True)

            midpoints = binned_ds.epoch_bins.to_index().mid
            binned_ds = binned_ds.assign_coords(epoch=('epoch_bins', midpoints))
            binned_ds = binned_ds.swap_dims({"epoch_bins":"epoch"}).drop_vars("epoch_bins")

            pa_flag = get_pA_flags(binned_ds).rename({'epoch': 'time'})

            # --- RUN NEW UNIFIED BAYESIAN PIPELINE (GROUPED) ---
            spc_anisotropy = calculate_spc_anisotropy(
                init_ds, uncert_map, 
                method='Bayesian_MCMC', 
                threshold=min_flux, 
                no_weights=no_weights, 
                groups=windows, 
                win_max=win_max, 
                use_avg_bfield=avg_bfield,
                plot_diagnostics=plot_diagnostics
            )

            # Group the support variables directly from the original init_ds using the same time bins
            support_binned = init_ds[support_vars].groupby_bins('epoch', fit_time).mean(skipna=True)
            support_binned = support_binned.assign_coords(epoch=('epoch_bins', midpoints))
            support_spc = support_binned.swap_dims({"epoch_bins":"epoch"}).drop_vars("epoch_bins").rename({'epoch': 'time'})
        else:
            # --- RUN NEW UNIFIED BAYESIAN PIPELINE (SINGLE EVENTS) ---
            spc_anisotropy = calculate_spc_anisotropy(
                init_ds, uncert_map, 
                method='Bayesian_MCMC', 
                threshold=min_flux, 
                no_weights=no_weights, 
                groups=None, 
                win_max=win_max, 
                use_avg_bfield=avg_bfield,
                plot_diagnostics=plot_diagnostics
            )

            support_spc = init_ds[support_vars].rename({'epoch' : 'time'})

        spc_anisotropy = xr.merge([spc_anisotropy, pa_flag, support_spc])

        # Build Filename
        tags = []
        if smooth_seconds:
            if isinstance(smooth_seconds, int):
                tags.append(f"{int(smooth_seconds)}s")
            elif smooth_seconds % 1 != 0:
                smooth_seconds_ms = int(smooth_seconds * 1000)
                tags.append(f"{smooth_seconds_ms}ms")

        if fit_seconds:
            tags.append(f"averaged_{fit_seconds}s")
        if no_weights:
            tags.append('nw')
        if win_max:
            tags.append('winmax')

        tag_str = "_" + "_".join(tags) if tags else ""
        output_dir = "./anisotropy_files"
        os.makedirs(output_dir, exist_ok=True)

        if avg_bfield:
            filename = f"{output_dir}/SPC_Anisotropy_{date_str}_{spc_mode}_bavg_{tag_str}"
        else:
            filename = f"{output_dir}/SPC_Anisotropy_{date_str}_{spc_mode}{tag_str}_Map_log_with_bias" #Final_public_data"# NEW_3PFIT_Bayesian_3D"

        print(f"Saving to {filename}...")
        cdflib.xarray.xarray_to_cdf(spc_anisotropy, filename)
        
    except Exception as e:
        print(f"!!! Error processing day {date_str} !!!")
        print(e)
        traceback.print_exc()
        
    finally:
        if 'init_ds' in locals(): del init_ds
        if 'spc_anisotropy' in locals(): del spc_anisotropy
        if 'subset' in locals(): del subset
        gc.collect()