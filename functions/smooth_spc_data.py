import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import flox

def smooth_spc_resample(ds, smooth_seconds, method='mean'):
    '''
    This function will smooth the spc data to smooth_second resolution.
    '''
    if 'mv_lo_dim' in ds.dims:
        ds = ds.rename({'mv_lo_dim' : 'mv_index'})

    # This function groups all data by unique times and mv_lo values.
    # Since mv_lo is decideded by the instruments commanding we have a set number of mv_lo values.
    ds_groupby = ds.groupby(['epoch', 'mv_lo']).mean(skipna=True)

    # Now we are going to resample the timeseries to the smooth seconds.
    # Note we use the right most label and subtract half of the time time resolution to get the mean value.
    if method.lower() == 'mean':
        ds_win_avg   = ds_groupby.resample(epoch=f'{smooth_seconds}s', label='right', origin='start', offset=f'{smooth_seconds/2}s').mean(skipna=True)
    if method.lower() == 'median':
        ds_win_avg = ds_groupby.resample(epoch=f'{smooth_seconds}s', label='right', origin='start', offset=f'{smooth_seconds/2}s').median(skipna=True)

    return ds_win_avg

def smooth_spc(ds, smooth_seconds):
    '''
    This function will smooth the spc data to smooth_second resolution.
    '''
    canonical_mv_lo = np.unique(ds.mv_lo.values)

    time_range_downsample = np.arange(ds.epoch.data[0],
                                      ds.epoch.data[-1],
                                      np.timedelta64(smooth_seconds, 's'))

    if len(time_range_downsample) % 2 == 0:
        locs = np.lib.stride_tricks.sliding_window_view(time_range_downsample, 2)
    else:
        time_range_downsample = np.concatenate((time_range_downsample, 
                                               np.array([time_range_downsample[-1] + np.timedelta64(smooth_seconds, 's')])), axis=0)
        locs = np.lib.stride_tricks.sliding_window_view(time_range_downsample, 2)

    reduced_list = []
    count = 0
    for tindx in tqdm(locs):
        # xarray.Dataset
        ds_temp = ds.sel(epoch=slice(tindx[0], tindx[1]))
        
        # n_orig = 32#len(ds_temp.mv_lo.v_dim)
        #print(l + 1020)
        if ds_temp.epoch.size > 0:
            timestamp =  np.datetime64(pd.to_datetime(tindx).mean(), 'ns')
            # group by energy channel/mv_lo values
            ds_e_avg = ds_temp.groupby('mv_lo').mean(skipna=True, engine='flox')
            ds_e_avg = ds_e_avg.reindex({'mv_lo' : canonical_mv_lo}, fill_value=np.nan)
            ds_e_avg = ds_e_avg.assign_coords({'time' : timestamp}).expand_dims('time')
            # num_to_add = n_orig - len(ds_e_avg.mv_lo)
            # n_temp = xr.full_like(ds_e_avg.isel(mv_lo=np.arange(num_to_add)), fill_value=np.nan)
            
            #reduced_list.append(ds_e_avg)
            # temp = xr.concat([ds_e_avg, n_temp], dim='mv_lo')
            # temp = temp.expand_dims('time')
            #reduced_list.append(temp)
        else:
            # Fill with nan values
            # temp = xr.full_like(ds.isel(time=0), fill_value=np.nan).swap_dims({'v_dim' : 'mv_lo'})
            timestamp = np.datetime64(tindx[0], 'ns') + np.timedelta64(int(1e9 * (smooth_seconds / 2)), 'ns')
            ds_e_avg = xr.full_like(ds.isel(epoch=0), fill_value=np.nan)
            ds_e_avg = ds_e_avg.assign_coords({'time': timestamp}).expand_dims('time')
            # temp = temp.assign_coords()
            # temp = temp.expand_dims('time')

        # temp = temp.reset_index(['mv_lo']).rename_dims({'mv_lo' : 'vdim'})

        reduced_list.append(ds_e_avg)
        count += 1
    return(reduced_list)

def smooth_spc_loop(ds, smooth_seconds):
    '''
    This function will smooth the spc data to smooth_second resolution.
    '''
    canonical_mv_lo = np.unique(ds.mv_lo.values)

    time_range_downsample = np.arange(ds.epoch.data[0],
                                      ds.epoch.data[-1],
                                      np.timedelta64(smooth_seconds, 's'))

    if len(time_range_downsample) % 2 == 0:
        locs = np.lib.stride_tricks.sliding_window_view(time_range_downsample, 2)
    else:
        time_range_downsample = np.concatenate((time_range_downsample, 
                                               np.array([time_range_downsample[-1] + np.timedelta64(smooth_seconds, 's')])), axis=0)
        locs = np.lib.stride_tricks.sliding_window_view(time_range_downsample, 2)

    reduced_list = []
    count = 0
    for tindx in tqdm(locs):
        # xarray.Dataset
        ds_temp = ds.sel(epoch=slice(tindx[0], tindx[1]))
        
        if ds_temp.epoch.size > 0:
            timestamp =  np.datetime64(pd.to_datetime(tindx).mean(), 'ns')
            # group by energy channel/mv_lo values
            ds_e_avg = ds_temp.groupby('mv_lo').mean(skipna=True, engine='flox')
            ds_e_avg = ds_e_avg.reindex({'mv_lo' : canonical_mv_lo}, fill_value=np.nan)
            ds_e_avg = ds_e_avg.assign_coords({'time' : timestamp}).expand_dims('time')
        else:
            # Fill with nan values
            timestamp = np.datetime64(tindx[0], 'ns') + np.timedelta64(int(1e9 * (smooth_seconds / 2)), 'ns')
            ds_e_avg = xr.full_like(ds.isel(epoch=0), fill_value=np.nan)
            ds_e_avg = ds_e_avg.assign_coords({'time': timestamp}).expand_dims('time')

        reduced_list.append(ds_e_avg)
        count += 1
    return(reduced_list)

def slow_smooth_spc(ds, smooth_seconds):
    '''
    This function will smooth the spc data to smooth_second resolution.
    '''
    minimumCadence = np.min(np.diff(ds.epoch.data))

    time_range_downsample = np.arange(ds.epoch.data[0],
                                      ds.epoch.data[-1],
                                      np.timedelta64(smooth_seconds, 's'))

    if len(time_range_downsample) % 2 == 0:
        locs = np.lib.stride_tricks.sliding_window_view(time_range_downsample, 2)
    else:
        time_range_downsample = np.concatenate((time_range_downsample, 
                                               np.array([time_range_downsample[-1] + np.timedelta64(smooth_seconds, 's')])), axis=0)
        locs = np.lib.stride_tricks.sliding_window_view(time_range_downsample, 2)

    reduced_list = []
    for tindx in tqdm(locs):
        # xarray.Dataset
        ds_temp = ds.sel(epoch=slice(tindx[0], tindx[1]))
        
        # 1D np.array
        mv_lo_temp = ds_temp.mv_lo.data
        mv_hi_temp = ds_temp.mv_hi.data

        if ds_temp.epoch.size > 0:
            timestamp = np.mean(ds_temp.epoch)
            
            mv_lo_new = np.full(mv_lo_temp.shape[1], np.nan)
            mv_hi_new = np.full(mv_lo_temp.shape[1], np.nan)
            
            valid_mv_lo_flag = np.where(~np.isnan(mv_lo_temp))

            mv_lo_valid = mv_lo_temp[valid_mv_lo_flag]

            # Define the unique mv_array
            unique_mv_lo_range = np.unique(np.sort(mv_lo_valid))
            
            ds_e_mean = []      # The mean of each energy channel
            count = 0
            # loop over each energy channel
            for mv in unique_mv_lo_range:
                r = np.where(mv_lo_temp == mv)

                # Take the mean of all values at the given energy channel
                ds_temp_mean = np.mean(ds_temp.isel(time=r[0], v_dim=r[1]))
                ds_temp_mean = ds_e_mean.append(ds_temp_mean)

                mv_lo_new[count] = mv
                mv_hi_new[count] = np.median(mv_hi_temp[r])

                count += 1

            num_to_add = 32 - count
            temp = xr.concat(ds_e_mean, dim='v_dim')

            n_temp = xr.full_like(temp.isel(v_dim=np.arange(num_to_add)), fill_value=np.nan)
            temp = xr.concat((temp, n_temp), dim='v_dim')

            temp = temp.assign_coords({'time': timestamp})

            mv_lo_new_da = xr.DataArray(mv_lo_new, name='mv_lo', dims='v_dim')
            mv_hi_new_da = xr.DataArray(mv_hi_new, name='mv_hi', dims='v_dim')
            
            temp = xr.merge([temp, mv_lo_new_da, mv_hi_new_da])
            
            reduced_list.append(temp)
        else:
            print('Here')
            # Fill with nan values
            temp = xr.full_like(ds.isel(time=0), fill_value=np.nan)
            timestamp = tindx[0] + np.timedelta64(500, 'ms')
            temp = temp.assign_coords({'time': timestamp})

            mv_lo_new_da = xr.DataArray(np.full(np.ones(32), np.nan), name='mv_lo', dims='v_dim')
            mv_hi_new_da = xr.DataArray(np.full(np.ones(32), np.nan), name='mv_hi', dims='v_dim')

            temp = xr.merge([temp, mv_lo_new_da, mv_hi_new_da])
            reduced_list.append(temp)

    return(reduced_list)