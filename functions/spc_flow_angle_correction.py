import numpy as np
import scipy as sp

def spc_flow_angle_correction(elevation, azimuth, xr_l3, GUESSES=False, FOV=80):
    # Get the points that we are correcting for
    # We want to correct the peak elevation and peak azimuth angles using the hot plasma 
    # correction table. 
    points = np.array([elevation, azimuth])

    # This is the plasma Mach number.
    if len(np.atleast_1d(xr_l3.time.data)) == 1:
        # data_mach_number = (np.linalg.norm(xr_l3.vp_fit_SC.data)/xr_l3.wp_fit.data)  
        data_mach_number = np.abs(xr_l3.vp_fit_SC.data[:,2])/xr_l3.wp_fit.data  
        data_mach_number = np.atleast_1d(data_mach_number)  
    else:
        # data_mach_number = (np.linalg.norm(xr_l3.vp_fit_SC.data, axis=1)/xr_l3.wp_fit.data)
        data_mach_number = np.abs(xr_l3.vp_fit_SC.data[:,2]/xr_l3.wp_fit.data)

    # Load in the lookup-table
    dict = sp.io.readsav('/home/michael/Research/pySolarWind/spp_swp_spc_calfiles_spc_uvphithetalookup_20200417.idl')

    # Load in the data table.
    diff_cd_ab = dict['table'][0][0]         # us
    diff_ad_bc = dict['table'][0][1]         # vs from Mike's pipeline script
    Mach_number = dict['table'][0][4]

    phi     = dict['table'][0][2]
    theta   = dict['table'][0][3]

    fov_mask = np.sqrt((-1.) * (np.sqrt(phi**2 + theta**2) > FOV))

    theta_guess = np.zeros(len(data_mach_number))
    phi_guess   = np.zeros(len(data_mach_number))
    theta_corr  = np.zeros(len(data_mach_number))
    phi_corr    = np.zeros(len(data_mach_number))

    for i in range(len(data_mach_number)):
        # Get the nearest mach number
        midx = np.argmin(np.abs(Mach_number - data_mach_number[i]))

        # Make sure this is the flow angle difference only. Not 0.5 * degrees2radians.
        if len(data_mach_number) == 1:
            thisu, thisv = (2.0*np.pi/180.) * points   # u ---> theta, v ---> phi    
        else:
            thisu, thisv = (2.0*np.pi/180.) * points[:,i]   # u ---> theta, v ---> phi

        resid = fov_mask + (diff_cd_ab[midx] - thisu)**2 + (diff_ad_bc[midx] - thisv)**2
        idxs = np.unravel_index(np.nanargmin(resid), shape=resid.shape)

        theta_guess[i], phi_guess[i] = theta[*idxs], phi[*idxs]

        # Calculate the local gradients
        dt = (theta[idxs[0] + 1, idxs[1]] - theta[idxs[0] - 1, idxs[1]])
        du = (diff_cd_ab[midx][idxs[0] + 1, idxs[1]] + diff_cd_ab[midx][idxs[0] - 1, idxs[1]])

        dt_du = dt/du

        dp = (phi[idxs[0], idxs[1] + 1] - phi[idxs[0], idxs[1] - 1])
        dv = (diff_cd_ab[midx][idxs[0], idxs[1] + 1] + diff_cd_ab[midx][idxs[0], idxs[1] - 1])

        dp_dv = dp/dv

        dt_corr = (thisu - diff_cd_ab[midx][*idxs]) * dt_du
        dp_corr = (thisv - diff_ad_bc[midx][*idxs]) * dp_dv

        if (np.isfinite(dt_corr) != 1) or (np.abs(dt_corr) > 0.25): dt_corr = 0
        if (np.isfinite(dp_corr) != 1) or (np.abs(dp_corr) > 0.25): dp_corr = 0

        theta_corr[i], phi_corr[i] = theta_guess[i] + dt_corr, phi_guess[i] + dp_corr

        # if dv or du == 0.0:
        #     if len(data_mach_number) == 1:
        #         theta_corr[i], phi_corr[i] = points
        #     else:
        #         theta_corr[i], phi_corr[i] = points[:,i]
    
    if GUESSES:
        return(theta_corr, phi_corr, theta_guess, phi_guess)
    

    return(theta_corr, phi_corr)


def spc_flow_angle_correction_FA(flow_angles, vp, wp, GUESSES=False, FOV=80):
    # Get the points that we are correcting for
    # We want to correct the peak elevation and peak azimuth angles using the hot plasma 
    # correction table. 
    points = flow_angles # This is a 2 x N array

    # This is the plasma Mach number.
    if len(np.atleast_1d(vp)) == 1:
        data_mach_number = np.abs(vp)/wp  
        if len(points[0]) == 1:
            data_mach_number = np.atleast_1d(data_mach_number)  
        else:
            data_mach_number = np.ones(len(points[0])) * data_mach_number
        
    else:
        # data_mach_number = (np.linalg.norm(xr_l3.vp_fit_SC.data, axis=1)/xr_l3.wp_fit.data)
        data_mach_number = np.abs(vp/wp)


    print('Mach Number', data_mach_number)

    # Load in the lookup-table
    dict = sp.io.readsav('/home/michael/Research/pySolarWind/spp_swp_spc_calfiles_spc_uvphithetalookup_20200417.idl')

    # Load in the data table.
    diff_cd_ab = dict['table'][0][0]         # us
    diff_ad_bc = dict['table'][0][1]         # vs from Mike's pipeline script
    Mach_number = dict['table'][0][4]

    phi     = dict['table'][0][2]
    theta   = dict['table'][0][3]

    fov_mask = np.sqrt((-1.) * (np.sqrt(phi**2 + theta**2) > FOV))

    theta_guess = np.zeros(len(data_mach_number))
    phi_guess   = np.zeros(len(data_mach_number))
    theta_corr  = np.zeros(len(data_mach_number))
    phi_corr    = np.zeros(len(data_mach_number))

    for i in range(len(data_mach_number)):
        # Get the nearest mach number
        midx = np.argmin(np.abs(Mach_number - data_mach_number[i]))

        # Make sure this is the flow angle difference only. Not 0.5 * degrees2radians.
        if len(data_mach_number) == 1:
            thisu, thisv = (2.0*np.pi/180.) * points   # u ---> theta, v ---> phi    
        else:
            thisu, thisv = (2.0*np.pi/180.) * points[:,i]   # u ---> theta, v ---> phi

        if np.isnan(thisu) or np.isnan(thisv):
            theta_corr[i], phi_corr[i] = np.nan, np.nan
            theta_guess[i], phi_guess[i] = np.nan, np.nan
            continue

        resid = fov_mask + (diff_cd_ab[midx] - thisu)**2 + (diff_ad_bc[midx] - thisv)**2
        idxs = np.unravel_index(np.nanargmin(resid), shape=resid.shape)

        theta_guess[i], phi_guess[i] = theta[*idxs], phi[*idxs]

        # Calculate the local gradients
        dt = (theta[idxs[0] + 1, idxs[1]] - theta[idxs[0] - 1, idxs[1]])
        du = (diff_cd_ab[midx][idxs[0] + 1, idxs[1]] + diff_cd_ab[midx][idxs[0] - 1, idxs[1]])

        dt_du = dt/du

        dp = (phi[idxs[0], idxs[1] + 1] - phi[idxs[0], idxs[1] - 1])
        dv = (diff_cd_ab[midx][idxs[0], idxs[1] + 1] + diff_cd_ab[midx][idxs[0], idxs[1] - 1])

        dp_dv = dp/dv

        dt_corr = (thisu - diff_cd_ab[midx][*idxs]) * dt_du
        dp_corr = (thisv - diff_ad_bc[midx][*idxs]) * dp_dv

        if (np.isfinite(dt_corr) != 1) or (np.abs(dt_corr) > 0.25): dt_corr = 0
        if (np.isfinite(dp_corr) != 1) or (np.abs(dp_corr) > 0.25): dp_corr = 0

        theta_corr[i], phi_corr[i] = theta_guess[i] + dt_corr, phi_guess[i] + dp_corr
    
    if GUESSES:
        return(theta_corr, phi_corr, theta_guess, phi_guess)

    return(theta_corr, phi_corr)




