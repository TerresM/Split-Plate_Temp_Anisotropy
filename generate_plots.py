import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import xarray as xr
import pandas as pd
import cdflib
import cdflib.epochs_astropy as epochs_astropy
import pyspedas
import datetime

import functions.span_flow_angle_calc as flow_angle

def Temperature_Anisotropy_Timeseries_Plot(spc_fit, span_fit, span_time, span_el, span_az, BOUNDS=None):
    """
    This function generates a timeseries plot for the supplied data.

    Parameters:
    -----------
    spc_fit : xarray.Dataset 
        Dataset containing all the spc fits
    span_fit : xarray.Dataset 
        Dataset containing all the span fits 
    """

    date_str = str(spc_vdf_fit.time.data[0])[0:10]

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)

    spec = gs.GridSpec(nrows=5, ncols = 1, figure=fig)

    ax1 = fig.add_subplot(spec[0:2])
    ax2 = fig.add_subplot(spec[2], sharex=ax1)
    ax3 = fig.add_subplot(spec[3], sharex=ax1)
    ax4 = fig.add_subplot(spec[4], sharex=ax1)

    axes = [ax1, ax2, ax3, ax4]

    for ax in axes[0:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    ylabels = [r'$T_{\perp}/T_{||}$', r'Elevation ($\theta$)', r'Azimuth ($\phi$)', 'B-Flow Angle']
    for i, ax in enumerate(axes):
        ax.set_ylabel(f"{ylabels[i]}", fontsize=14)

    ax1.set_yscale('log')
    ax1.plot(span_fit.time, span_fit.Trat1, color='tab:red', linewidth=1, label='SPAN-i')
    ax1.plot(spc_fit.time, spc_fit.T_ani_avg, color='k', linewidth=1, label='SPC')

    ax1.fill_between(spc_fit.time, spc_fit.T_ani_avg + spc_fit.T_ani_avg_delta_plus, spc_fit.T_ani_avg - spc_fit.T_ani_avg_delta_minus, color='grey', alpha=0.3)
    
    ax1.set_ylim([0.1, 30])
    ax1.legend(ncols=2, fontsize=20)

    ax2.plot(spc_fit.time, spc_fit.peak_elevation, color='k')
    ax2.plot(span_time, span_el, linewidth=1, color='tab:red', label='SPAN-i')

    ax3.plot(spc_fit.time, spc_fit.peak_azimuth, color='k')
    ax3.plot(span_time, span_az, linewidth=1, color='tab:red', label='SPAN-i')

    ax4.plot(spc_fit.time, spc_fit.B_x_flow_angle, color='tab:green', label='Bx-angle')
    ax4.plot(spc_fit.time, spc_fit.B_y_flow_angle, color='tab:blue', label='By-angle')  
    ax4.axhline(0, linestyle='dotted', color='k')
    ax4.axhline(90, linestyle='dashed', color='k')  
    ax4.axhline(-90, linestyle='dashed', color='k')  
    ax4.legend()

    if BOUNDS:
        ax1.set_xlim([BOUNDS[0], BOUNDS[1]])
        bound_str1 = str(bounds[0])[-18:-10]
        bound_str2 = str(bounds[1])[-18:-10]
        plt.savefig(f'./figures/{date_str}_{bound_str1}_{bound_str2}.png')
    else:
        plt.savefig(f'./figures/{date_str}_full.png')

    plt.show()

def Tani_Timeseries_Plot(spc_fit, span_fit, span_time, T_FA_RAT, span_el, span_az, BOUNDS=None):
    """
    This function generates a timeseries plot for the supplied data.

    Parameters:
    -----------
    spc_fit : xarray.Dataset 
        Dataset containing all the spc fits
    span_fit : xarray.Dataset 
        Dataset containing all the span fits 
    """

    date_str = str(spc_vdf_fit.time.data[0])[0:10]

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)

    spec = gs.GridSpec(nrows=5, ncols = 1, figure=fig)

    ax1 = fig.add_subplot(spec[0:2])
    ax2 = fig.add_subplot(spec[2], sharex=ax1)
    ax3 = fig.add_subplot(spec[3], sharex=ax1)
    ax4 = fig.add_subplot(spec[4], sharex=ax1)

    axes = [ax1, ax2, ax3, ax4]

    for ax in axes[0:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    ylabels = [r'$T_{\perp}/T_{||}$', r'Elevation ($\theta$)', r'Azimuth ($\phi$)', 'B-Flow Angle']
    for i, ax in enumerate(axes):
        ax.set_ylabel(f"{ylabels[i]}", fontsize=14)

    ax1.set_yscale('log')
    # ax1.plot(span_fit.time, span_fit.Trat1, color='tab:red', linewidth=1, alpha=0.7, label='Bi-Max Fit')
    ax1.plot(span_time, T_FA_RAT, color='tab:blue', linewidth=1, alpha=0.7, label='SPAN-i Moment')
    ax1.plot(spc_fit.time, spc_fit.T_ani_avg, color='k', linewidth=1, alpha=0.7, label='SPC')
    print(T_FA_RAT)
    ax1.fill_between(spc_fit.time, spc_fit.T_ani_avg + spc_fit.T_ani_avg_delta_plus, spc_fit.T_ani_avg - spc_fit.T_ani_avg_delta_minus, color='grey', alpha=0.3)
    
    ax1.set_ylim([0.1, 30])
    ax1.legend(ncols=3, fontsize=20)

    ax2.plot(spc_fit.time, spc_fit.peak_elevation, color='k', label='SPC')
    ax2.plot(span_time, span_el, linewidth=1, color='tab:red', label='SPAN-i')
    ax2.fill_between(span_time, span_el - 5., span_el + 5., color='tab:red', alpha=0.4)
    ax2.set_ylim([-60,60])
    ax2.legend()

    ax3.plot(spc_fit.time, spc_fit.peak_azimuth, color='k')
    ax3.plot(span_time, span_az, linewidth=1, color='tab:red', label='SPAN-i')
    ax3.fill_between(span_time, span_az - 5., span_az + 5., color='tab:red', alpha=0.4)
    ax3.set_ylim([-60,60])

    ax4.plot(spc_fit.time, spc_fit.B_x_flow_angle, color='tab:green', label='Bx-angle')
    ax4.plot(spc_fit.time, spc_fit.B_y_flow_angle, color='tab:blue', label='By-angle')  
    ax4.axhline(0, linestyle='dotted', color='k')
    ax4.axhline(90, linestyle='dashed', color='k')  
    ax4.axhline(-90, linestyle='dashed', color='k')  
    ax4.legend()
    ax4.set_ylim([-90,90])

    if BOUNDS:
        ax1.set_xlim([BOUNDS[0], BOUNDS[1]])
        bound_str1 = str(bounds[0])[-18:-10]
        bound_str2 = str(bounds[1])[-18:-10]
        plt.savefig(f'./figures/{date_str}_{bound_str1}_{bound_str2}.png')
    else:
        plt.savefig(f'./figures/{date_str}_full.png')

    plt.show()

def Badman_T_anisotropy(Txx, Tzz, b_vec):
    # This function calculates the temperature anisotropy using Badman_et_al_2025
    '''
    Parameters:
    -----------
    t_tens : ndarray
        The symmetric temperature tensor. 
    b_vec : ndarray
        The magnetic field vector. 
    Outputs:
    --------
    t_fa : ndarray
        Temperature tensor in field aligned coordinates
    '''
    magb = np.linalg.norm(b_vec, axis=1)
    theta = np.arccos(b_vec[:,2]/magb)

    T_perp = Tzz + (Txx - Tzz)/(1 - ((b_vec[:,0]/magb)**2) * np.tan(theta)**2)
    T_para = T_perp + (Txx - Tzz)/(((b_vec[:,0]/magb)**2) * np.sin(theta)**2 - np.cos(theta)**2)

    T_fa = np.vstack([T_perp, T_para])

    return(T_fa)

if __name__ == "__main__":
    tstart = datetime.datetime(2024, 9, 26)
    tend   = datetime.datetime(2024, 9, 27)
    trange = [tstart, tend]

    psp_span = pyspedas.psp.spi(trange, datatype='sf00_l3_mom', level='l3', downloadonly=True, last_version=True)

    span_data_l3 = xr.concat([cdflib.xarray.cdf_to_xarray(f, to_datetime=False) for f in psp_span], dim='Epoch', data_vars=['DENS', 'VEL_INST', 'VEL_SC', 'T_TENSOR_INST', 'TEMP', 'MAGF_INST', 'MAGF_SC'])

    # Get the temperature anisotropy.
    T_TENS = span_data_l3.T_TENSOR_INST.data
    T_xx = T_TENS[:,0]
    T_zz = T_TENS[:,2]

    T_FA = Badman_T_anisotropy(T_xx, T_zz, span_data_l3.MAGF_INST.data)

    span_time_object = epochs_astropy.CDFAstropy.convert_to_astropy(span_data_l3.Epoch.data)
    span_time = span_time_object.utc.datetime       # Ensure we are in utc

    mom_dataset = span_data_l3.copy()
    mom_dataset['Epoch'] = span_time

    span_vel_inst = span_data_l3.VEL_INST.data

    span_el, span_az = flow_angle.flow_angles_span_inst(span_vel_inst)

    # fits = pd.concat(map(pd.read_csv, ['/home/michael/Research/pySolarWind/spp_swp_spi_sf00_2023-09-24.csv', 
    #                                '/home/michael/Research/pySolarWind/spp_swp_spi_sf00_2023-09-25.csv', 
    #                                '/home/michael/Research/pySolarWind/spp_swp_spi_sf00_2023-09-26.csv']), ignore_index=True)
    fits = pd.concat(map(pd.read_csv, ['/home/michael/Research/pySolarWind/spp_swp_spi_sf00_2023-09-26.csv']), ignore_index=True)
    fits['time'] = pd.to_datetime(fits['time'], unit='s')

    xr_fits = fits.to_xarray()

    names = list(xr_fits.data_vars.keys())
    names = [i for i in names if i != 'time']
    values = list(xr_fits.data_vars.values())

    temp = []
    for i, name in enumerate(names):
        DA = xr.DataArray(values[i+1].data, dims=['time'], coords=dict(time = xr_fits.time.data))
        temp.append(DA)

    data_dict = {name : data_array for name, data_array in zip(names, temp)}
    fit_dataset = xr.Dataset(data_dict)


    # spc_fit_files = ['./Fit_Data/2023-09-26_xr_Fit_v2.cdf']
    spc_fit_files = [f'./Fit_Data/Fixed_RDF/{str(np.datetime64(tstart))[:10]}_00:00:00_to_23:59:59_xr_Fit_v2.cdf']

    spc_vdf_fit = xr.concat([cdflib.xarray.cdf_to_xarray(f, to_datetime=True) for f in spc_fit_files], dim='time')

    #bounds = [spc_vdf_fit.time[0].data, spc_vdf_fit.time[len(spc_vdf_fit.time.data)//2].data]
    bounds = [tstart, tend]

    # Temperature_Anisotropy_Timeseries_Plot(spc_vdf_fit, fit_dataset, span_time, span_el, span_az, bounds)
    Tani_Timeseries_Plot(spc_vdf_fit, None, span_time, T_FA[1]/T_FA[0], span_el, span_az, bounds)

    