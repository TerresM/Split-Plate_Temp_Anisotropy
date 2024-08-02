import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cdflib

from scipy.optimize import curve_fit

def linear_model(x, a, b):
    """Simple linear model"""
    return a * x + b

def find_slopes(model, xdata, ydata, weights, method='lm'):
    """
    This function calculates the anisotropy assuming a linear fit.

    Parameters:
    -----------
    model : func
        This is assumed to be a linear model y = mx + b. This is left to be 
        general for now.
    xdata : numpy.array
        Represents the x-axis data. 
    ydata : numpy.array
        Represents the y-axis data.
    weights : numpy.array
        Represent the weights.
    
    Return:
    -------
    popt : numpy array
        Results of the linear fitting.
    pcov : numpy.ndarray Matrix
        Result from the curve-fit. Used to calculate the standard deviation of the error.
    lienar_fit : numpy array
        Result of the linear fit
    r_sq : float
        R-squared statistic
    red_chi_sq : float
        reduced chi-sqaured statistic
    slope_err : float
        Error in slope
    std_dev_err : float
        Given by np.sqrt(np.diag(pcov)).
    """
    # Perform the fitting, using scipy
    popt, pcov = curve_fit(model, xdata, ydata, sigma=weights, method=method)

    # Build the linear fit for the xdata and the corresponding slope and y-intercept
    linear_fit = model(xdata, *popt)

    # Calculate the residual variances i.e. difference of data to the model. 
    residual_variance = (ydata - linear_fit)**2

    # Sum of ydata - ymodel/ymodel
    chi_sq = np.sum(residual_variance / linear_fit)

    # The reduced chi squared value. divide by N-2!
    red_chi_sq = chi_sq / (len(xdata) - 2)

    # Variance of the data!
    total_variance = (ydata - np.mean(ydata))**2

    # 1 - model_variance / data_variance
    r_sq = 1 - (np.sum(residual_variance) / np.sum(total_variance))**2

    # Fun way to say variance in x!
    abscissa_variance = (xdata  - np.mean(xdata))**2
    slope_err = np.sqrt(np.sum(residual_variance) / np.sum(abscissa_variance)) / (len(xdata) - 2)

    return(popt, pcov, linear_fit, r_sq, red_chi_sq, slope_err)

def _spc_nan_anisotropy(time_fill, FILLVAL=np.nan):
    """
    This function is purely defined for computational convinences of using xarray.
    This is likely not the optimal way to structure this function however, it is simple.

    Parameters:
    -----------
    time_fill : np.datetime64
        The timestamp that failed the initial requirements for anisotorpy fitting.
    FILLVAL : any
        Defined value that all variables will be assigned in the xarray.Dataset
    Return:
    -------
    results : xarray.Dataset
        Full dataset containing all FILLVALS
    """

    results = xr.Dataset(
        data_vars=dict(
            T_ani_avg = FILLVAL,
            T_ani_avg_uncert = FILLVAL,
            T_ani_avg_delta_plus = FILLVAL,
            T_ani_avg_delta_minus = FILLVAL,
            T_ani_chi =  FILLVAL,
            T_ani_chi_uncert =  FILLVAL,
            T_ani_chi_delta_plus =  FILLVAL,
            T_ani_chi_delta_minus =  FILLVAL,
            T_ani_psi =  FILLVAL,
            T_ani_psi_uncert =  FILLVAL,
            T_ani_psi_delta_plus =  FILLVAL,
            T_ani_psi_delta_minus =  FILLVAL,
            slope_chi =  FILLVAL,
            slope_psi =  FILLVAL,
            intercept_chi =  FILLVAL,
            intercept_psi =  FILLVAL,
            r_sq_chi =  FILLVAL,
            r_sq_psi =  FILLVAL,
            chi_sq_chi =  FILLVAL,
            chi_sq_psi =  FILLVAL,
            slope_err_chi =  FILLVAL,
            slope_err_psi =  FILLVAL,
            Nx =  FILLVAL,
            Ny =  FILLVAL,
            Nz =  FILLVAL,
            B_x_flow_angle = FILLVAL,
            B_y_flow_angle = FILLVAL,
            peak_azimuth = FILLVAL,
            peak_elevation = FILLVAL
        ),
        coords = dict(
            time = time_fill
        ),
        attrs = dict(Note = f"Fitting Results for {time_fill - np.timedelta64(2, 's')} to {time_fill + np.timedelta64(2, 's')}")
    )

    return(results)

def _calc_anisotropy_from_fits(fit_value, NZ):
    '''
    This function takes the linear fit value for the slope and returns
    the corresponding anisotropy value. This comes from solving for alpha from 

    chi = -(alpha - 1)Bz / ((alpha - 1)(1 - Nz^2) + 1)

    Solving for alpha 

    alpha = -(chi / (chi (1 - Nz^2) + Nz))

    Parameters:
    -----------
    fit_value : the slope found from the linear regression.

    NZ : value of the magnetic field unit vector in the z-direction.

    Returns:
    --------
    alpha : the raw value of the temperature anisotorpy

    T_ani : sqrt(alpha)**2, to remove nan and negative anisotropy values!
    '''
    alpha = (-fit_value[0]/(fit_value[0] * (1. - NZ**2) + NZ)) + 1

    if np.isfinite(alpha) and (alpha > 0):
        T_ani = np.sqrt(alpha)**2
    else:
        T_ani = np.nan
    
    return(alpha, T_ani)

def _get_uncertainty(fit_vals, slope_error, Nz, field_error, flow_angle, calc_anisotropy, uncertainty_map):
    '''
    This function will calculate the upper and lower bound uncertainties from 
    a simulated ion VDF response to the given functions. 
    
    Parameters:
    ----------
    fit_vals : np.array
        Array containing the linear fits from temperature anisotropy measurement.
    slope_error : float
        Value corresponding to the error in the corresponding fit values.
    Nz : float
        Magnetic field normal direction in SPC frame.
    field_error : float
        Error associated with magnetic field measurements. Assumed to be 1 nT.
    flow_angle : np.array
        Magnetic field flow angle.
    calc_anisotropy : float
        Calculated anisotropy value form the given linear fit
    uncertainty_map : np.array
        Contains the lookup table information for the corresponding uncertainties.

    Return:
    -------
    delta_plus : float
        Upper uncertainty value
    delta_minus : float
        Lower Uncertainty value
    std : float
        Standard deviation of measurement
    delta_mag :
        magnitude of the upper and lower delta
    '''
    # First load in the uncertainty maps
    angle_list, ani_list, anisotropy_delta_plus, anisotropy_delta_minus = uncertainty_map

    if np.isnan(flow_angle):
        mapped_angle_index = -1    # Is this correct? Perhaps I just set the uncertainty to NaN?
    else:
        mapped_angle_index = np.nanargmin(np.abs(flow_angle - angle_list))

    if np.isnan(calc_anisotropy):
        mapped_anisotropy_index = -1 
    else:
        mapped_anisotropy_index = np.nanargmin(np.abs(calc_anisotropy - ani_list))

    delta_plus  = anisotropy_delta_plus[mapped_angle_index, mapped_anisotropy_index]
    delta_minus = anisotropy_delta_minus[mapped_angle_index, mapped_anisotropy_index]

    # Now get the standard deviation
    part_tani_part_slope = -Nz / ((1 - Nz**2) * fit_vals[0] + Nz)**2
    part_tani_part_Nz_chi = (fit_vals[0] - 2. * Nz * fit_vals[0]**2)/ ((1 - Nz**2) * fit_vals[0] + Nz)**2 

    std = np.sqrt((part_tani_part_slope * slope_error)**2 + (part_tani_part_Nz_chi * field_error)**2)

    delta_mag = np.sqrt(delta_plus**2 + delta_minus**2 + std**2)

    return(delta_plus, delta_minus, std, delta_mag)

def spc_calculate_anisotropy(timestamp, ds_spc, ds_fields, uncertainty_map, 
                             threshold=0.5, PEAK_NORM=False, TOTAL_CURRENT=False,
                             NO_WEIGHTS=False):
    ''' 
    This function does all the work in calculating the temperature anisotropy.
    The method is defined as in Paulson et al. 2024 [in preparation]. 

    Parameters:
    -----------
    timestamp : np.datetime64
        The average time for the given window.
    ds_spc : xarray.Dataset
        Contains all the SPC data with the derived reduced distribution function
    ds_fields : xarray.Dataset
        Contains the 1-minute averaged magnetic field
    uncertainty_map : np.ndarray
        Multidimensional array containing a lookup table for uncertainty based on 
        magnetic field flow angles. 

    Keywords:
    ---------
    PEAK_NORM : Boolean (default : False)
        Determines the normalization of the vdfs. 
            - PEAK_NORM = True ===> Normalize all VDFs in window to max vdf value.
            - PEAK_NORM = False ===> Normalize each VDF to unity.
    TOTAL_CURRENT : Boolean (default : False)
        Flag that will use total current for weighting in place of Reduced VDF.

    NO_WEIGHTS : Boolean (default : False)
        If true return fitting with no weights applied.            

    Return:
    -------
    results : xarray.Dataset
        Contains the derived temperature anisotropy and supporting variables
    '''
    # Get the average flow angles
    peak_azimuth = np.nanmean(ds_spc.peak_azimuth.data)
    peak_elevation = np.nanmean(ds_spc.peak_elevation.data)

    # Grab the necessary data from the spc data set
    vx = ds_spc.vel_x.data
    vy = ds_spc.vel_y.data

    # Note there is a negative value on vz to affirm with PSP convention
    vz = -ds_spc.vel_mid.data

    # Define the unit vector of the background magnetic field. 

    # Check if the fields data has only 1 element. If it does, our calculation is slightly different.
    if ds_fields.ndim == 1:
        N_vec = ds_fields.data/np.linalg.norm(ds_fields.data)
        Nx_calc, Ny_calc, Nz_calc = N_vec

        # Dot product of vz and Nhat
        vz_bx = vz * N_vec[0]
        vz_by = vz * N_vec[1]
    else:
        N_vec = ds_fields.data/np.linalg.norm(ds_fields.data, axis=1)[:,None]
        Nx_calc, Ny_calc, Nz_calc = np.mean(N_vec, axis=0)

        # Dot product of vz and Nhat
        vz_bx = vz * N_vec[:,0, None]
        vz_by = vz * N_vec[:,1, None]

    # Now we need to define the weights. 

    if PEAK_NORM:
        if TOTAL_CURRENT:
            # Take the total current and normalize to the peak total current 
            # in given window
            weights = ds_spc.total_current.data/np.nanmax(ds_spc.total_current.data)
        else:
            weights = ds_spc.rdf.data/np.nanmax(ds_spc.rdf.data)
    else:
        if TOTAL_CURRENT:
            weights = ds_spc.total_current.data/np.nanmax(ds_spc.total_current.data, axis=1)[:,None]
        else:
            weights = ds_spc.rdf.data/np.nanmax(ds_spc.rdf.data, axis=1)[:,None]

    # Now let us define a mask for our weights.
    # NOTE: we use the flatten here as the mask will only work for a 1D array in 
    # this implementaiton.
    mask_weights = np.where(weights.flatten() > threshold)[0]

    # Now let us define all the 1D arrays needed for the linear fitting
    weight_fits = weights.flatten()[mask_weights]
    vz_fit      = vz.flatten()[mask_weights]    # Not used in our fits, however, is helpful for plotting

    vx_fit      = vx.flatten()[mask_weights]
    vzbx_fit    = vz_bx.flatten()[mask_weights]

    vy_fit      = vy.flatten()[mask_weights]
    vzby_fit    = vz_by.flatten()[mask_weights]

    if NO_WEIGHTS:
        weight_fits = np.ones(len(weight_fits))

    # Fixed the above code by making a new function
    if len(vx_fit) < 2:
        results = _spc_nan_anisotropy(timestamp)
        return(results)

    # Perform the linear fits    
    params_chi, cov_chi, chi_linear_fit, r_sq_chi, chi_sq_chi, slope_err_chi = find_slopes(linear_model, vzbx_fit, vx_fit, 1./weight_fits, 'lm')
    params_psi, cov_psi, psi_linear_fit, r_sq_psi, chi_sq_psi, slope_err_psi = find_slopes(linear_model, vzby_fit, vy_fit, 1./weight_fits, 'lm')

    # Get the anisotropy values!
    alpha_chi, Tani_chi = _calc_anisotropy_from_fits(params_chi, Nz_calc)
    alpha_psi, Tani_psi = _calc_anisotropy_from_fits(params_psi, Nz_calc)

    # Define the magnetic field flow angles
    xangle = np.degrees(np.arctan(Nx_calc / Nz_calc))
    yangle = np.degrees(np.arctan(Ny_calc / Nz_calc))

    # Now we do the propogation of errors to get the average anisotorpy values.
    chi_delta_plus, chi_delta_minus, chi_std, chi_delta_mag = _get_uncertainty(params_chi, slope_err_chi, Nz_calc, 1, xangle, Tani_chi, uncertainty_map)
    psi_delta_plus, psi_delta_minus, psi_std, psi_delta_mag = _get_uncertainty(params_psi, slope_err_psi, Nz_calc, 1, yangle, Tani_psi, uncertainty_map)

    combined_delta_mag = np.sqrt(chi_delta_mag**2 + psi_delta_mag**2)

    weight_chi = 1./(chi_delta_mag / combined_delta_mag)
    weight_psi = 1./(psi_delta_mag / combined_delta_mag)

    # Explicitly define the values that we need. 
    # NOTE: This could be much more compact. However, it is clear what is happening.
    combined_uncert             = np.sqrt((chi_delta_mag**2 + psi_delta_mag**2))/2
    combined_uncert_delta_plus  = np.sqrt((chi_delta_plus + psi_delta_plus**2))/2
    combined_uncert_delta_minus = np.sqrt((chi_delta_minus**2 + psi_delta_minus**2))/2

    chi_anisotropy = Tani_chi
    chi_anisotropy_uncertainty = chi_delta_mag
    chi_anisotropy_uncertainty_delta_plus = chi_delta_plus
    chi_anisotropy_uncertainty_delta_minus = chi_delta_minus
    chi_anisotropy_fit_uncertainty = chi_std**2

    psi_anisotropy = Tani_psi
    psi_anisotropy_uncertainty = psi_delta_mag
    psi_anisotropy_uncertainty_delta_plus = psi_delta_plus
    psi_anisotropy_uncertainty_delta_minus = psi_delta_minus
    psi_anisotropy_fit_uncertainty = psi_std**2


    # At this point we need to check if Tani_chi and Tani_psi exist
    if np.isnan(Tani_chi) and np.isnan(Tani_psi):
        average_anisotropy = np.nan
        average_anisotropy_uncertainty = np.nan
        average_anisotropy_uncertainty_delta_plus  = np.nan
        average_anisotropy_uncertainty_delta_minus = np.nan
        average_anisotropy_fit_uncertainty = np.nan
    elif np.isnan(Tani_chi):
        average_anisotropy = Tani_psi
        average_anisotropy_uncertainty = psi_delta_mag
        average_anisotropy_uncertainty_delta_plus = psi_delta_plus
        average_anisotropy_uncertainty_delta_minus = psi_delta_minus
        average_anisotropy_fit_uncertainty = psi_std**2
    elif np.isnan(Tani_psi):
        average_anisotropy = Tani_chi
        average_anisotropy_uncertainty = chi_delta_mag
        average_anisotropy_uncertainty_delta_plus = chi_delta_plus
        average_anisotropy_uncertainty_delta_minus = chi_delta_minus
        average_anisotropy_fit_uncertainty = chi_std**2
    else:
        average_anisotropy = (weight_chi * Tani_chi + weight_psi * Tani_psi) / (weight_chi + weight_psi)
        average_anisotropy_uncertainty = combined_uncert + (np.abs(Tani_chi - Tani_psi)/ np.sqrt(2))
        average_anisotropy_uncertainty_delta_plus = np.sqrt(combined_uncert_delta_plus**2 + np.abs(Tani_chi - Tani_psi)**2 / 2.) #np.sqrt((chi_delta_plus**2 + psi_delta_plus**2)/4 + (np.abs(Tani_chi - Tani_psi)**2 / 2.))
        average_anisotropy_uncertainty_delta_minus = np.sqrt(combined_uncert_delta_minus**2 + np.abs(Tani_chi - Tani_psi)**2 / 2.) #np.sqrt((chi_delta_minus**2 + psi_delta_minus**2)/4 + (np.abs(Tani_chi - Tani_psi)**2 / 2.))
        average_anisotropy_fit_uncertainty = np.sqrt(chi_std**2 + psi_std**2)

    results = xr.Dataset(
        data_vars=dict(
            T_ani_avg = average_anisotropy,
            T_ani_avg_uncert = average_anisotropy_uncertainty,
            T_ani_avg_delta_plus = average_anisotropy_uncertainty_delta_plus,
            T_ani_avg_delta_minus = average_anisotropy_uncertainty_delta_minus,
            T_ani_chi =  chi_anisotropy,
            T_ani_chi_uncert =  chi_anisotropy_uncertainty,
            T_ani_chi_delta_plus =  chi_anisotropy_uncertainty_delta_plus,
            T_ani_chi_delta_minus =  chi_anisotropy_uncertainty_delta_minus,
            T_ani_psi =  psi_anisotropy,
            T_ani_psi_uncert =  psi_anisotropy_uncertainty,
            T_ani_psi_delta_plus =  psi_anisotropy_uncertainty_delta_plus,
            T_ani_psi_delta_minus =  psi_anisotropy_uncertainty_delta_minus,
            slope_chi =  params_chi[0],
            slope_psi =  params_psi[0],
            intercept_chi =  params_chi[1],
            intercept_psi =  params_psi[1],
            r_sq_chi =  r_sq_chi,
            r_sq_psi =  r_sq_psi,
            chi_sq_chi =  chi_sq_chi,
            chi_sq_psi =  chi_sq_psi,
            slope_err_chi =  slope_err_chi,
            slope_err_psi =  slope_err_psi,
            Nx =  Nx_calc,
            Ny =  Ny_calc,
            Nz =  Nz_calc,
            B_x_flow_angle = np.degrees(np.arctan(Nx_calc/Nz_calc)),
            B_y_flow_angle = np.degrees(np.arctan(Ny_calc/Nz_calc)),
            peak_azimuth = peak_azimuth,
            peak_elevation = peak_elevation
        ),
        coords = dict(
            time = timestamp
        ),
        attrs = dict(Note = f"Fitting Results for {ds_spc.time.data[0]} to {ds_spc.time.data[-1]}")
    )

    return(results)