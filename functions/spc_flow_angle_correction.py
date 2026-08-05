import numpy as np
import scipy as sp
import numpy as np
import scipy as sp
from numba import njit, prange
from scipy.spatial import cKDTree

@njit(parallel=True, fastmath=True)
def _apply_gradients_numba_1d(elev_flat, azim_flat, midx_flat, best_x_flat, best_y_flat,
                              diff_cd_ab, diff_ad_bc, theta, phi):
    N = len(elev_flat)
    Nx, Ny = theta.shape

    theta_corr = np.full(N, np.nan)
    phi_corr = np.full(N, np.nan)
    theta_guess = np.full(N, np.nan)
    phi_guess = np.full(N, np.nan)

    for i in prange(N):
        midx = midx_flat[i]
        bx = best_x_flat[i]
        by = best_y_flat[i]

        if midx < 0 or bx == -1 or by == -1:
            continue

        thisu = (2.0 * np.pi / 180.0) * elev_flat[i]
        thisv = (2.0 * np.pi / 180.0) * azim_flat[i]

        t_guess = theta[bx, by]
        p_guess = phi[bx, by]
        theta_guess[i] = t_guess
        phi_guess[i] = p_guess

        if (bx > 0 and bx < Nx - 1) and (by > 0 and by < Ny - 1):
            dt = theta[bx + 1, by] - theta[bx - 1, by]
            du = diff_cd_ab[midx, bx + 1, by] - diff_cd_ab[midx, bx - 1, by]
            dt_du = dt / du if du != 0 else 0.0

            dp = phi[bx, by + 1] - phi[bx, by - 1]
            dv = diff_ad_bc[midx, bx, by + 1] - diff_ad_bc[midx, bx, by - 1]
            dp_dv = dp / dv if dv != 0 else 0.0

            dt_corr = (thisu - diff_cd_ab[midx, bx, by]) * dt_du
            dp_corr = (thisv - diff_ad_bc[midx, bx, by]) * dp_dv

            if np.isnan(dt_corr) or abs(dt_corr) > 0.25: dt_corr = 0.0
            if np.isnan(dp_corr) or abs(dp_corr) > 0.25: dp_corr = 0.0

            theta_corr[i] = t_guess + dt_corr
            phi_corr[i] = p_guess + dp_corr
        else:
            theta_corr[i] = t_guess
            phi_corr[i] = p_guess

    return theta_corr, phi_corr, theta_guess, phi_guess

def spc_flow_angle_correction_1d_fast(elev_1d, azim_1d, mach_1d, sav_file_path, FOV=80):
    """
    Expects strictly finite 1D arrays. 
    Returns the corrected 1D arrays.
    """
    N = len(elev_1d)
    
    # Load IDL data
    sav_data = sp.io.readsav(sav_file_path)
    diff_cd_ab = np.array(sav_data['table'][0][0], dtype=np.float64)
    diff_ad_bc = np.array(sav_data['table'][0][1], dtype=np.float64)
    phi = np.array(sav_data['table'][0][2], dtype=np.float64)
    theta = np.array(sav_data['table'][0][3], dtype=np.float64)
    Mach_table = np.array(sav_data['table'][0][4], dtype=np.float64)
    
    Nx, Ny = theta.shape
    base_fov_mask = np.sqrt(phi**2 + theta**2) <= FOV

    # Map points to Mach Index
    mach_diffs = np.abs(mach_1d[:, None] - Mach_table[None, :])
    midx_1d = np.argmin(mach_diffs, axis=1)

    best_x_1d = np.full(N, -1, dtype=np.int32)
    best_y_1d = np.full(N, -1, dtype=np.int32)

    target_u = (2.0 * np.pi / 180.0) * elev_1d
    target_v = (2.0 * np.pi / 180.0) * azim_1d
    
    unique_midxs = np.unique(midx_1d)

    # KD-Tree search
    for midx in unique_midxs:
        u_raw = diff_cd_ab[midx]
        v_raw = diff_ad_bc[midx]
        
        finite_table_mask = base_fov_mask & np.isfinite(u_raw) & np.isfinite(v_raw)
        finite_grid_indices = np.arange(Nx * Ny)[finite_table_mask.flatten()]
        
        tree_points = np.column_stack((u_raw[finite_table_mask], v_raw[finite_table_mask]))
        kdtree = cKDTree(tree_points)

        query_mask = (midx_1d == midx)
        queries = np.column_stack((target_u[query_mask], target_v[query_mask]))
        
        _, nearest_idx = kdtree.query(queries, workers=-1) 

        x_idx, y_idx = np.unravel_index(finite_grid_indices[nearest_idx], (Nx, Ny))
        best_x_1d[query_mask] = x_idx
        best_y_1d[query_mask] = y_idx

    # Apply Numba Gradients (using the _apply_gradients_numba_1d function from earlier)
    t_corr, p_corr, _, _ = _apply_gradients_numba_1d(
        elev_1d, azim_1d, midx_1d, best_x_1d, best_y_1d, diff_cd_ab, diff_ad_bc, theta, phi
    )

    return t_corr, p_corr

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
        dv = (diff_cd_ab[midx][idxs[0], idxs[1] + 1] + diff_ad_bc[midx][idxs[0], idxs[1] - 1])

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




