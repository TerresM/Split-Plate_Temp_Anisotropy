import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import cdflib
plt.ion()

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

import astropy.constants as c 
import astropy.units as u 
import astropy
import pandas as pd

from tqdm import tqdm

import pyspedas, pytplot

import datetime

import json

# Import user defined functions
from functions.data_preprocessing import clean_variable as clean
from functions.spc_current_to_rdf_v2 import derive_reduced_vdf

from support_functions.gen_spc_uncert_map_2d import gen_uncertainty_map
from functions.spc_do_anisotropy_calc import perform_fitting_with_time_windows


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def median_filter_current(a_current, b_current, c_current, d_current):
    current_stack = np.stack([a_current, b_current, c_current, d_current])

    # set all fillvals to NaNs
    current_stack[current_stack < 0] = np.nan

    median_current = np.median(current_stack, axis=0)

    for i in range(4):
        current_stack[i][current_stack[i] > 3*median_current] = median_current[current_stack[i] > 3*median_current]

    return(current_stack)

def plot_spc_with_bfield(spc_data, flds_data):
    fig, ax = plt.subplots(2, figsize=(16, 8), layout='constrained')
    ax[0].scatter(np.repeat(spc_data.time.data, 32).reshape(spc_data.vel_mid.shape), np.nan_to_num(spc_data.vel_mid, -1e31), c=spc_data.rdf, norm=LogNorm(1e-2, 5), cmap='inferno', marker='s', s=30)
    ax[0].set_yscale('log')

    ax[1].plot(flds_data.time, flds_data.data[:,0], linewidth=3, label=r'$B_x$')
    ax[1].plot(flds_data.time, flds_data.data[:,1], linewidth=3, label=r'$B_y$')
    ax[1].plot(flds_data.time, flds_data.data[:,2], linewidth=3, label=r'$B_z$')
    ax[1].legend()

    plt.show()

if __name__ == "__main__":
    # Set the time range that we are interested in
    tstart = datetime.datetime(2023, 9, 26)
    tend   = datetime.datetime(2023, 9, 27)
    trange = [tstart, tend]
    trange_bfield = [tstart - datetime.timedelta(days= 1), tend + datetime.timedelta(days= 1)]

    date_str = str(tstart)[0:10] 

    # Load in the Downsampled 1s SPC data from KP compression
    data_ds = cdflib.cdf_to_xarray(f'{sys.path[0]}/SPC_DownsampledSpectra_{date_str}_1sec.cdf', to_datetime=True)

    angle_bins = 4  # degrees
    ani_bins = 30   # N-bins 0.03 to 15

    uncert_map = gen_uncertainty_map(angle_bins, ani_bins)

    credentials = load_config('config.json')

    
    psp_bfield = pyspedas.psp.fields(trange_bfield,
                                     datatype='mag_SC_1min',
                                     level='l2',
                                     notplot=False,
                                     get_support_data=True,
                                     last_version=True,
                                     username=credentials['psp']['fields']['username'],
                                     password=credentials['psp']['fields']['password'])
    
    psp_spc_l3  = pyspedas.psp.spc(trange=trange,
                                   datatype='l3i',
                                   level='L3',
                                   notplot=True,
                                   downloadonly=True,
                                   get_support_data=True,
                                   last_version=True,
                                   username=credentials['psp']['sweap']['username'],
                                   password=credentials['psp']['sweap']['password'])

    psp_span_l3 = pyspedas.psp.spi(trange=trange,
                                   datatype='spi_sf00',
                                   level='L3',
                                   notplot=True,
                                   downloadonly=True,
                                   get_support_data=True,
                                   last_version=True,
                                   username=credentials['psp']['sweap']['username'],
                                   password=credentials['psp']['sweap']['password'])

    spc_l3 = cdflib.cdf_to_xarray(psp_spc_l3[0], to_datetime=True, fillval_to_nan=True).rename({'Epoch' : 'time'})

    # Drop full-scan values
    flag_is_1 = np.squeeze(np.argwhere(spc_l3.general_flag.data == 1))
    spc_l3_peak = spc_l3.drop_isel(time=flag_is_1)


    data_ds.mv_lo.data[data_ds.mv_lo.data == -1] = np.nan
    data_ds.mv_hi.data[data_ds.mv_hi.data == -1] = np.nan

    med_filt_current = median_filter_current(data_ds.spc_a_mean.data, data_ds.spc_b_mean.data, data_ds.spc_c_mean.data, data_ds.spc_d_mean.data)

    spc = xr.Dataset(
        data_vars=dict(
            mv_lo = (['time', 'dim'], data_ds.mv_lo.data),
            mv_hi = (['time', 'dim'], data_ds.mv_hi.data),
            a_current = (['time', 'dim'], med_filt_current[0]),   #data_ds.spc_a_mean.data),
            b_current = (['time', 'dim'], med_filt_current[1]),   #data_ds.spc_b_mean.data),
            c_current = (['time', 'dim'], med_filt_current[2]),   #data_ds.spc_c_mean.data),
            d_current = (['time', 'dim'], med_filt_current[3]),   #data_ds.spc_d_mean.data),
        ),
        coords=dict(
            time = data_ds.epoch.data,
            dim = np.arange(32)
        ),
        attrs=dict(Note = "Selected Data Quantities")
    )
    
    # Set bad_voltage cutoff
    minvoltage = 150    # Volts
    maxvoltage = 20000   # Volts

    minflux = 0.5       # pA

    bad_voltage_points_lo = (spc.mv_lo.data < minvoltage)
    bad_voltage_points_hi = (spc.mv_lo.data > maxvoltage)

    bad_voltage_points = bad_voltage_points_lo | bad_voltage_points_hi

    # Identify the bad flux points
    # aflux = spc.a_current.data <= minflux
    # bflux = spc.b_current.data <= minflux
    # cflux = spc.c_current.data <= minflux
    # dflux = spc.d_current.data <= minflux

    # bad_flux = aflux | bflux | cflux | dflux

    total_current = (spc.a_current + spc.b_current + spc.c_current + spc.d_current).data
    bad_flux = total_current < minflux

    # Drop all the points where the bad voltage mask is met!
    for vname in spc.data_vars:
        var = spc[vname]
        if 'time' in var.dims and 'dim' in var.dims:
            var.data[bad_voltage_points] = np.nan
            var.data[bad_flux] = np.nan

    # Drop periods with more than 12 of the 32 energy channels missing.
    nan_count = np.sum(np.isnan(spc.mv_lo.data), axis=1)
    spc = spc.drop_isel(time=np.where(nan_count > 20))

    spc_sel = spc       #.sel(time=slice(subint1, subint2))

    # TODO: Consider making my own function to do this smoothing after!
    # NOTE: I have already implemented a Box-Car windowing average.

    # Convert to get the Reduced VDF. Still using random factor to define the geometric factor!
    spc_rdf = derive_reduced_vdf(spc_sel)
    
    # Now that the SPC has been delt with. let us fix the bfield
    if 'psp_fld_l2_mag_SC_4_Sa_per_Cyc' in pytplot.tnames():
        fields = clean(pytplot.data_quants['psp_fld_l2_mag_SC_4_Sa_per_Cyc'])
    else:    
        fields = clean(pytplot.data_quants['psp_fld_l2_mag_SC_1min'])

    # No longer need to use the subint values!
    flds_sel = fields   
    flds_1s  = fields.interp(time=spc_rdf.time)

    # Interpolate the l3 Peak track data to the same time
    spc_l3_1s = spc_l3_peak.interp(time=spc.time, method='linear')
    
    # Now let us group over 4 second windows. 
    fit_seconds = 4
    fit_steps = np.timedelta64(tend - tstart)/np.timedelta64(fit_seconds, 's')
    fit_time  = np.datetime64(tstart) + np.arange(fit_steps+1)*np.timedelta64(fit_seconds, 's')

    windows = np.squeeze(np.dstack([fit_time[0:-1:1], fit_time[1::1]]))

    # TODO: Fix the temp values. This is supposed to give the l3 fit and moment values averaged over each window.
    # this step is not necessarily needed however, is conducted as a check. This needs to be fixed. 
    values, temp = perform_fitting_with_time_windows(windows, spc_rdf, spc_l3_1s, flds_1s, 
                                                        uncert_map, VERBOSE=True, AS_XARRAY=True,
                                                        SUPPORT=True, PEAK_NORM=True, TOTAL_CURRENT=False,
                                                        NO_WEIGHTS=False, PLOT=False)    

    result_xarray = values

    # Save our file.
    cdflib.xarray_to_cdf(result_xarray, f'/home/michael/Research/pySolarWind/PSP/Fit_Data/Fixed_RDF/{date_str}_xr_Fit_v2.cdf')

    sys.exit()


    # Make everything into an array
    T_ani_avg = result_xarray.T_ani_avg #np.array(T_ani_avg)
    T_dp = result_xarray.T_ani_avg_delta_plus #np.array(T_dp)
    T_dm = result_xarray.T_ani_avg_delta_minus #np.array(T_dm)

    T_ani_psi = result_xarray.T_ani_psi #np.array(T_ani_psi)
    T_ani_chi = result_xarray.T_ani_chi #np.array(T_ani_chi)


    np1 = dataset['np1']
    
    vp1_x = dataset['vp1_x']
    vp1_y = dataset['vp1_y']
    vp1_z = dataset['vp1_z']

    vp_mag_span = np.sqrt(vp1_x**2 + vp1_y**2 + vp1_z**2)

    T_perp  = dataset['Tperp1']
    
    T_ratio = dataset['Trat1']
    T_para  = T_perp / T_ratio

    T_total = T_perp + T_para

    t = dataset['time']
    time = [datetime.datetime.utcfromtimestamp(f) for f in np.array(t)]
    
    time_numpy = np.array([np.datetime64(f, 'ns') for f in time])

    # Let us now get the SPC data that we need to have calculated
    spc_4s = spc_rdf.resample(time='4S').mean(dim='time')

    # Load spc L3 data
    vp1_spc = clean(pytplot.data_quants['spp_spc_vp1_fit_SC'])
    wp1_spc = clean(pytplot.data_quants['spp_spc_wp1_fit'])

    # Now reduce the SPC L3 data to the 4s resolution and take the mean
    vp1_spc_4s = vp1_spc.sel(time=slice(subint1, subint2)).resample(time='4S').mean()
    wp1_spc_4s = wp1_spc.sel(time=slice(subint1, subint2)).resample(time='4S').mean()

    t_spc_temp = ((c.m_p * (wp1_spc_4s.data**2) * (u.km/u.s)**2)/c.k_B).to('K')
    t_spc_eV   = t_spc_temp * 8.617e-5

    ts_spc = (spc_4s.time.data - np.datetime64('1975-01-01T00:00:00'))/np.timedelta64(1, 's')

    ts_span = (time_numpy - np.datetime64('1975-01-01T00:00:00'))/np.timedelta64(1, 's')

    # Interpolate 
    T_ani_span = np.interp(ts_spc, ts_span, T_ratio)
    v_span_x = np.interp(ts_spc, ts_span, vp1_x)
    v_span_y = np.interp(ts_spc, ts_span, vp1_y)
    v_span_z = np.interp(ts_spc, ts_span, vp1_z)

    v_span_mag = np.sqrt(v_span_x**2 + v_span_y**2 + v_span_z**2)
    T_span_tot = np.interp(ts_spc, ts_span, T_total)

    cdflib.xarray_to_cdf(result_xarray, f'/home/michael/Research/pySolarWind/PSP/Encounter_Investigation/E17/spc_anisotropy_fit_{date_str}_v01.cdf')

    PLOTTING = False
    if PLOTTING:
        fig, ax = plt.subplots(3, figsize=(12, 8), sharex=True, layout='constrained')

        ax[0].plot(flds_1s.time.data, flds_1s.data[:,0], color='tab:red', label=r'$B_x$')
        ax[0].plot(flds_1s.time.data, flds_1s.data[:,1], color='tab:green', label=r'$B_y$')
        ax[0].plot(flds_1s.time.data, flds_1s.data[:,2], color='tab:blue', label=r'$B_z$')
        ax[0].legend(ncols=3)

        ax[1].plot(time, T_ratio, color='tab:blue', linewidth=1, label='SPAN-i')
        ax[1].plot(result_xarray.time, T_ani_avg, color='k', linewidth=1, label='SPC')
        ax[1].fill_between(result_xarray.time, T_ani_avg - T_dp, T_ani_avg + T_dp, color='grey', alpha=0.4, label='Uncertainty')

        ax[0].set_xlim([result_xarray.time.data[0], result_xarray.time.data[-1]])
        ax[1].set_xlim([result_xarray.time.data[0], result_xarray.time.data[-1]])

        ax[1].set_yscale('log')

        ax[0].set_ylabel('1-min Average \n Magnetic Field [nT]', fontsize=20)
        ax[1].set_ylabel(r'$T_{\perp} / T_{||}$', fontsize=20)

        ax[2].plot(result_xarray.time.data, result_xarray.B_x_flow_angle.data, label=r'$\Theta_{Bx}$', color='tab:red')
        ax[2].plot(result_xarray.time.data, result_xarray.B_y_flow_angle.data, label=r'$\Theta_{By}$', color='tab:blue')
        ax[2].set_ylim([-90, 90])
        ax[2].axhline(0, linestyle='dashed', color='k', linewidth=2)
        ax[2].axhline(45, linestyle='dashed', color='k', linewidth=2, alpha=0.3)
        ax[2].axhline(-45, linestyle='dashed', color='k', linewidth=2, alpha=0.3)
        ax[2].set_ylabel('Degrees', fontsize=20)
        ax[2].legend(ncols=2, fontsize=14)


        fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True, layout='constrained')

        ax[0].plot(time, T_ratio, color='tab:blue', linewidth=1, label='SPAN')
        ax[0].plot(result_xarray.time, T_ani_avg, color='k', linewidth=1, label='Avg')
        ax[0].plot(result_xarray.time, T_ani_chi, color='tab:orange', linewidth=1, label=r'$\chi$')
        ax[0].plot(result_xarray.time, T_ani_psi, color='tab:green', linewidth=1, label=r'$\psi$')

        ax[0].set_yscale('log')
        ax[0].set_ylabel(r'$T_{\perp} / T_{||}$', fontsize=20)
        ax[0].set_xlabel(r'Time [UT]')
        ax[0].set_xlim([result_xarray.time.data[0], result_xarray.time.data[-1]])

        ax[1].plot(result_xarray.time.data, result_xarray.B_x_flow_angle.data, label=r'$\Theta_{Bx}$', color='tab:red')
        ax[1].plot(result_xarray.time.data, result_xarray.B_y_flow_angle.data, label=r'$\Theta_{By}$', color='tab:blue')
        ax[1].set_ylim([-90, 90])
        ax[1].axhline(0, linestyle='dashed', color='k', linewidth=2)
        ax[1].axhline(45, linestyle='dashed', color='k', linewidth=2, alpha=0.3)
        ax[1].axhline(-45, linestyle='dashed', color='k', linewidth=2, alpha=0.3)
        ax[1].set_ylabel('Degrees', fontsize=20)
        ax[1].legend(ncols=2, fontsize=14)
                        




        fig, ax = plt.subplots(2, figsize=(12,8), sharex=True, layout='constrained')
        ax[0].plot(result_xarray.time, np.linalg.norm(vp1_spc_4s, axis=1), color='k')
        ax[0].plot(spc_4s.time, v_span_mag, color='r', alpha=0.5)

        ax[0].set_ylabel('|V|')
        ax[0].set_yscale('log')

        ax[1].plot(spc_4s.time, t_spc_eV, color='k')
        ax[1].plot(spc_4s.time, T_span_tot, color='r', alpha=0.5)

        ax[1].set_yscale('log')
        ax[1].set_ylabel('T (eV)')


        # Let us now define a few metrics to calculate how close everything is to the actual values.
        vel_diff = np.abs(np.linalg.norm(vp1_spc_4s, axis=1) - v_span_mag)
        t_diff = np.abs(t_spc_eV.value - T_span_tot)

        vel_percent_diff = vel_diff/(0.5*(np.linalg.norm(vp1_spc_4s, axis=1) + v_span_mag))*100
        t_percent_diff = t_diff/(0.5*(t_spc_eV.value + T_span_tot))*100

        # Now lets take points that are within 10 % of each other.
        mask_v = vel_percent_diff < 10
        mask_t = t_percent_diff < 10

        mask_tot = mask_v & mask_t




        fig, ax = plt.subplots(5, figsize=(12, 8), sharex=True, layout='constrained')

        ax[0].plot(flds_1s.time.data, flds_1s.data[:,0], color='tab:red', label=r'$B_x$')
        ax[0].plot(flds_1s.time.data, flds_1s.data[:,1], color='tab:green', label=r'$B_y$')
        ax[0].plot(flds_1s.time.data, flds_1s.data[:,2], color='tab:blue', label=r'$B_z$')
        ax[0].legend(ncols=3)

        ax[1].plot(result_xarray.time, T_ani_span, color='tab:blue', linewidth=1, label='SPAN-i')
        ax[1].plot(result_xarray.time, T_ani_avg, color='k', linewidth=1, label='SPC')
        ax[1].fill_between(result_xarray.time, T_ani_avg - T_dp, T_ani_avg + T_dp, color='grey', alpha=0.4, label='Uncertainty')

        ax[0].set_xlim([result_xarray.time.data[0], result_xarray.time.data[-1]])
        ax[1].set_xlim([result_xarray.time.data[0], result_xarray.time.data[-1]])

        ax[1].set_yscale('log')

        ax[0].set_ylabel('1-min Average \n Magnetic Field [nT]', fontsize=10)
        ax[1].set_ylabel(r'$T_{\perp} / T_{||}$', fontsize=10)

        ax[2].plot(result_xarray.time, T_ani_span, color='tab:blue', linewidth=1, label='SPAN-i')
        ax[2].plot(result_xarray.time, T_ani_chi, color='k', linewidth=1, label=r'$\chi$-Fit SPC')
        ax[2].fill_between(result_xarray.time, T_ani_chi - result_xarray.T_ani_chi_delta_minus, T_ani_chi + result_xarray.T_ani_chi_delta_plus, color='grey', alpha=0.4, label='Uncertainty')
        ax[2].legend(ncols=3)
        ax[2].set_yscale('log')
                    
        ax[3].plot(result_xarray.time, T_ani_span, color='tab:blue', linewidth=1, label='SPAN-i')
        ax[3].plot(result_xarray.time, T_ani_psi, color='k', linewidth=1, label=r'$\psi$-Fit SPC')
        ax[3].fill_between(result_xarray.time, T_ani_psi - result_xarray.T_ani_psi_delta_minus, T_ani_psi + result_xarray.T_ani_psi_delta_plus, color='grey', alpha=0.4, label='Uncertainty')
        ax[3].legend(ncols=3)
        ax[3].set_yscale('log')

        ax[4].plot(result_xarray.time.data, result_xarray.B_x_flow_angle.data, label=r'$\Theta_{Bx}$', color='tab:red')
        ax[4].plot(result_xarray.time.data, result_xarray.B_y_flow_angle.data, label=r'$\Theta_{By}$', color='tab:blue')
        ax[4].set_ylim([-90, 90])
        ax[4].axhline(0, linestyle='dashed', color='k', linewidth=2)
        ax[4].axhline(45, linestyle='dashed', color='k', linewidth=2, alpha=0.3)
        ax[4].axhline(-45, linestyle='dashed', color='k', linewidth=2, alpha=0.3)
        ax[4].set_ylabel('Degrees', fontsize=20)
        ax[4].legend(ncols=2, fontsize=14)

