import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
import datetime

import pyspedas
import pytplot
import cdflib

import astropy.units as u 
import astropy.constants as c 

import json

from functions.data_preprocessing import clean_variable as clean



def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def spani_peak(eflux):
    '''
    This function locates the peak of the span ion dataset.
    '''
    data = eflux.data

    # if only one time is passed.
    if data.ndim == 1:
        idx = np.nanargmax(data)
    if data.ndim == 2:
        idx = np.nanargmax(data, axis=1)
    # if the data is passed in 8 x 32 x 8 for single time
    if data.ndim == 3:
        idx = np.nanargmax(data.flatten())
    if data.ndim == 4:
        idx = np.nanargmax(data.reshape(data.shape[0], -1), axis=1)

    pidx, eidx, tidx = np.unravel_index(idx, (8, 32, 8))

    return(pidx, eidx, tidx)

def flag_max_vdf_on_edge(eflux):
    '''
    Check if the maximum vdf falls within the FOV or is on the edge.
    
    Parameters:
    -----------
    eflux : str or np.array
        The differential energy flux. Which is a proxy for the ion vdf.
    '''
    pidx, eidx, tidx = spani_peak(eflux)
    
    # Define our flag
    flags = np.zeros(len(pidx))

    # Check if PHI values on edge
    flags[np.logical_or(pidx == 0, pidx == 7)] = 2  # Fill value
    
    # Check if THETA on edge
    flags[np.logical_or(tidx == 0, tidx == 7)] = 2  # Fill value

    return(flags)

def good_data_time(theta, eflux):
    '''
    Makes sure that the data in the period has the full field of view!
    ''' 
    keep = np.isfinite(theta)

    flag = []
    for i in range(keep.shape[0]):
        keep_sum = np.sum(keep[i, :])
        if keep_sum != 8 * 32 * 8:
            flag.append(2)
        else:
            flag.append(0)

    flag = np.array(flag)
    return(flag)


if __name__ == "__main__":
    credentials = load_config('config.json')

    perihelion = datetime.datetime(2023,9,27)
    
    tstart = perihelion - datetime.timedelta(days=5)
    tend   = perihelion + datetime.timedelta(days=5)

    trange = [tstart, tend]

    #psp_span_l2 = pyspedas.psp.spi(trange=trange, 
                                # datatype='spi_sf00', 
                                # level='L2', 
                                # notplot=False,
                                # get_support_data=True,
                                # last_version=True,
                                # username=credentials['psp']['sweap']['username'],
                                # password=credentials['psp']['sweap']['password'])
    
    psp_span_l3 = pyspedas.psp.spi(trange=trange,
                                    datatype='spi_sf00',
                                    level='L3',
                                    notplot=False,
                                    get_support_data=True,
                                    last_version=True,
                                    username=credentials['psp']['sweap']['username'],
                                    password=credentials['psp']['sweap']['password'])
    
    psp_spc_l3 = pyspedas.psp.spc(trange,
                            datatype='l3i',
                            level='L3',
                            notplot=False,
                            get_support_data=True,
                            last_version=True,
                            #no_update=True,
                            username=credentials['psp']['sweap']['username'],
                            password=credentials['psp']['sweap']['password'])
    
    psp_bfield = pyspedas.psp.fields(trange,
                                     datatype='mag_SC_4_Sa_per_Cyc',
                                     level='l2',
                                     notplot=False,
                                     get_support_data=True,
                                     last_version=True,
                                     #no_update=True,
                                     username=credentials['psp']['fields']['username'],
                                     password=credentials['psp']['fields']['password'])
    
    vp_fits = clean(pytplot.data_quants['spp_spc_vp_fit_SC'])
    general_flag = pytplot.data_quants['spp_spc_general_flag']

    B = clean(pytplot.data_quants['psp_spi_MAGF_SC'])

    # Choose only values where general flag is good
    flag_mask = np.where(general_flag.data == 0)[0]

    vp_fit_selected = vp_fits.isel(time=flag_mask)

    # L3 SPAN Data
    v_spi_mom = clean(pytplot.data_quants['psp_spi_VEL_SC'])

    # Energy Flux for data flags
    eflux_vs_energy = pytplot.data_quants['psp_spi_EFLUX_VS_ENERGY']
    eflux_vs_theta = pytplot.data_quants['psp_spi_EFLUX_VS_THETA']
    eflux_vs_phi = pytplot.data_quants['psp_spi_EFLUX_VS_PHI']

    # Check phi for FOV
    phi_sel = eflux_vs_phi.isel(v_dim=np.arange(7) + 1)

    # Check if values are less than max
    # phi_flag1 = np.max(phi_sel.data, axis=1) > eflux_vs_phi.data[:,0]
    # phi_flag2 = np.max(phi_sel.data, axis=1) > eflux_vs_phi.data[:,-1]

    # fov_flag = phi_flag1 & phi_flag2
    
    # Alternate flag
    peak_index = np.argmax(eflux_vs_phi.data, axis=1)
    peak_flag = ((peak_index != 0) & (peak_index != 7)).astype(int)

    

    fig, ax = plt.subplots(5, figsize=(12, 8), layout='constrained', sharex=True)

    ax[0].plot(eflux_vs_energy.time[np.where(peak_flag == 1)], peak_flag[np.where(peak_flag == 1)], marker='x', linestyle='none', markersize=0.6)
    ax[0].set_ylabel('SPAN FOV - FLAG')

    ax[1].plot(vp_fit_selected.time, vp_fit_selected.data[:,0], color='b', marker='.', linestyle='none', markersize=0.8, label='SPC')
    ax[1].plot(v_spi_mom.time, v_spi_mom.data[:,0], color='k', alpha=0.6, label='SPAN')
    ax[1].set_ylabel(r'$V_{x, sc}$')

    ax[2].plot(vp_fit_selected.time, vp_fit_selected.data[:,1], color='b', marker='.', linestyle='none', markersize=0.8, label='SPC')
    ax[2].plot(v_spi_mom.time, v_spi_mom.data[:,1], color='k', alpha=0.6, label='SPAN')
    ax[2].set_ylabel(r'$V_{y, sc}$')

    ax[3].plot(vp_fit_selected.time, vp_fit_selected.data[:,2], color='b', marker='.', linestyle='none', markersize=0.8, label='SPC')
    ax[3].plot(v_spi_mom.time, v_spi_mom.data[:,2], color='k', alpha=0.6, label='SPAN')
    ax[3].set_ylabel(r'$V_{z, sc}$')

    ax[4].plot(B.time, B.data[:,0], label='x')
    ax[4].plot(B.time, B.data[:,1], label='y')
    ax[4].plot(B.time, B.data[:,2], label='z')
    ax[4].set_ylabel(r'nT')
    ax[4].set_xlabel('Time')

    for i in range(5):
        ax[4].legend()