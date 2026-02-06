import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from tqdm import tqdm
from scipy.optimize import curve_fit

def get_data(obj, key):
    '''
    Helper function that delineates between dictonary and xarray.
    '''
    val = obj[key]

    if isinstance(val, xr.DataArray):
        return val.data
    return val

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
        mapped_angle_index = np.nanargmin(np.abs(np.abs(flow_angle) - angle_list))

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

def compute_anisotropy(params_chi, slope_err_chi, Tani_chi,
                       params_psi, slope_err_psi, Tani_psi,
                       Nz_calc, xangle, yangle, uncertainty_map):

    # helper to unpack uncertainty for one direction
    def anisotropy_uncertainty(params, slope_err, angle, Tani):
        delta_plus, delta_minus, std, delta_mag = _get_uncertainty(
            params, slope_err, Nz_calc, 1, angle, Tani, uncertainty_map
        )
        return {
            "anisotropy": Tani,
            "uncertainty": delta_mag,
            "uncertainty_delta_plus": delta_plus,
            "uncertainty_delta_minus": delta_minus,
            "fit_uncertainty": std**2,
        }

    chi = anisotropy_uncertainty(params_chi, slope_err_chi, xangle, Tani_chi)
    psi = anisotropy_uncertainty(params_psi, slope_err_psi, yangle, Tani_psi)

    # combined magnitudes
    combined_delta_mag = np.sqrt(chi["uncertainty"]**2 + psi["uncertainty"]**2)
    weights = {
        "chi": 1.0 / (chi["uncertainty"] / combined_delta_mag),
        "psi": 1.0 / (psi["uncertainty"] / combined_delta_mag),
    }

    combined_uncert = np.sqrt(chi["uncertainty"]**2 + psi["uncertainty"]**2) / 2
    combined_uncert_delta_plus = np.sqrt((chi["uncertainty_delta_plus"]**2 + psi["uncertainty_delta_plus"]**2)) / 2
    combined_uncert_delta_minus = np.sqrt((chi["uncertainty_delta_minus"]**2 + psi["uncertainty_delta_minus"]**2)) / 2

    # handle NaN cases
    if np.isnan(chi["anisotropy"]) and np.isnan(psi["anisotropy"]):
        avg = {k: np.nan for k in [
            "anisotropy", "uncertainty", "uncertainty_delta_plus",
            "uncertainty_delta_minus", "fit_uncertainty"
        ]}
    elif np.isnan(chi["anisotropy"]):
        avg = psi
    elif np.isnan(psi["anisotropy"]):
        avg = chi
    else:
        diff = np.abs(chi["anisotropy"] - psi["anisotropy"])
        avg = {
            "anisotropy": (weights["chi"] * chi["anisotropy"] + weights["psi"] * psi["anisotropy"]) /
                          (weights["chi"] + weights["psi"]),
            "uncertainty": combined_uncert + diff / np.sqrt(2),
            "uncertainty_delta_plus": np.sqrt(combined_uncert_delta_plus**2 + diff**2 / 2.0),
            "uncertainty_delta_minus": np.sqrt(combined_uncert_delta_minus**2 + diff**2 / 2.0),
            "fit_uncertainty": np.sqrt(chi["fit_uncertainty"] + psi["fit_uncertainty"]),
        }

    return chi, psi, avg

def compute_anisotropy_results(params_chi, slope_err_chi, Tani_chi,
                               params_psi, slope_err_psi, Tani_psi,
                               Nz_calc, xangle, yangle, uncertainty_map,
                               Nx_calc, Ny_calc, r_sq_chi, r_sq_psi,
                               chi_sq_chi, chi_sq_psi, peak_azimuth,
                               peak_elevation, timestamp, ds, Bvec):

    chi, psi, avg = compute_anisotropy(
        params_chi, slope_err_chi, Tani_chi,
        params_psi, slope_err_psi, Tani_psi,
        Nz_calc, xangle, yangle, uncertainty_map
    )

    # package everything in one dict
    data_vars = dict(
        T_ani_avg             = avg["anisotropy"],
        T_ani_avg_uncert      = avg["uncertainty"],
        T_ani_avg_delta_plus  = avg["uncertainty_delta_plus"],
        T_ani_avg_delta_minus = avg["uncertainty_delta_minus"],

        T_ani_chi             = chi["anisotropy"],
        T_ani_chi_uncert      = chi["uncertainty"],
        T_ani_chi_delta_plus  = chi["uncertainty_delta_plus"],
        T_ani_chi_delta_minus = chi["uncertainty_delta_minus"],

        T_ani_psi             = psi["anisotropy"],
        T_ani_psi_uncert      = psi["uncertainty"],
        T_ani_psi_delta_plus  = psi["uncertainty_delta_plus"],
        T_ani_psi_delta_minus = psi["uncertainty_delta_minus"],

        slope_chi      = params_chi[0],
        slope_psi      = params_psi[0],
        intercept_chi  = params_chi[1],
        intercept_psi  = params_psi[1],
        r_sq_chi       = r_sq_chi,
        r_sq_psi       = r_sq_psi,
        chi_sq_chi     = chi_sq_chi,
        chi_sq_psi     = chi_sq_psi,
        slope_err_chi  = slope_err_chi,
        slope_err_psi  = slope_err_psi,

        Nx = Nx_calc,
        Ny = Ny_calc,
        Nz = Nz_calc,

        Bx = Bvec[0],
        By = Bvec[1],
        Bz = Bvec[2],

        B_x_flow_angle = np.degrees(np.arctan(Nx_calc / Nz_calc)),
        B_y_flow_angle = np.degrees(np.arctan(Ny_calc / Nz_calc)),

        peak_azimuth   = peak_azimuth,
        peak_elevation = peak_elevation
    )

    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(time=[timestamp]),
        attrs=dict(Note=f"Fitting Results for {get_data(ds, 'epoch')[0]} to {get_data(ds, 'epoch')[-1]}")
    )

# def get_support_values(ds):
#     time = get_data(ds, 'epoch')

#     support_scalar = [
#         "np_fit", "np_moment",
#         "wp_fit", "wp_moment"
#     ]
#     support_vector = [
#         "vp_fit_SC", "vp_moment_SC"
#     ]

#     support_results = []
#     for i in tqdm(range(len(time))):
#         scalars = {v: get_data(ds, v) for v in support_scalar}
#         vectors = {v: (("vector_index",), get_data(ds, v)) for v in support_vector}
#         results = scalars | vectors
#         support_results.append(results)

#     return(xr.concat(support_results, dim='epoch'))

def get_support_values(ds):
    """
    Extracts support quantities (e.g., np_fit, wp_moment, vp_fit_SC)
    from the main dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset, which may contain support quantities.

    Returns
    -------
    xr.Dataset
        A new dataset containing only the found support variables.
    """

    support_scalar = [
        "np_fit", "np_moment",
        "wp_fit", "wp_moment"
    ]
    support_vector = [
        "vp_fit_SC", "vp_moment_SC"
    ]

    all_support_vars = support_scalar + support_vector

    # Find which of these variables actually exist in the dataset
    vars_to_keep = []
    for var_name in all_support_vars:
        if var_name in ds.data_vars:
            vars_to_keep.append(var_name)

    # Return a new dataset containing only those variables
    return ds[vars_to_keep]

def calculate_support_means_groups(ds, groups):
    """
    Compute time-averaged support quantities (e.g., np_fit, wp_moment, vp_fit_SC)
    over user-defined time groups.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing support quantities.
    groups : list of tuple
        Each element is (start, stop) np.datetime64 or pd.Timestamp.

    Returns
    -------
    xr.Dataset
        Dataset of group-averaged support variables, indexed by midpoint time.
    """

    support_scalar = [
        "np_fit", "np_moment",
        "wp_fit", "wp_moment"
    ]
    support_vector = [
        "vp_fit_SC", "vp_moment_SC"
    ]

    support_results = []

    for (t0, t1) in tqdm(groups, desc="Averaging support data"):
        block = ds.sel(epoch=slice(t0, t1))

        # Skip empty groups
        if block.epoch.size == 0:
            mid_time = (t0 + (t1 - t0) / 2)
            nan_scalars = {v: np.nan for v in support_scalar}
            nan_vectors = {v: (("vector_index",), np.full(ds[v].shape[1], np.nan)) for v in support_vector}
            nan_vars = nan_scalars | nan_vectors
            support_results.append(xr.Dataset(nan_vars, coords={"epoch": [mid_time]}))
            continue

        # Compute group means
        mean_vars = {}
        for v in support_scalar:
            mean_vars[v] = block[v].mean(dim="epoch", skipna=True).data
        for v in support_vector:
            mean_vars[v] = (("vector_index",), block[v].mean(dim="epoch", skipna=True).data)

        mid_time = (t0 + (t1 - t0) / 2)
        mean_ds = xr.Dataset(mean_vars, coords={"epoch": [mid_time]})
        support_results.append(mean_ds)

    return xr.concat(support_results, dim="epoch", coords='minimal')


def spc_calculate_anisotropy(ds, uncertainty_map, threshold = 0.5, NO_WEIGHTS=False):
    """
    This function does all the work for calculating the temperature anisotropy
    for Parker Solar Probe's Solar Probe Cup. The method is defined in the 
    Paulson et al. 2025. See attached notes for the derivation.

    Parameters:
    -----------
    ds : xarray.Dataset or dictonary
        Dataset containing current measurements for all collector plates, the downsampled
        magnetic field measurements, and the reduced distribution function already derived.

        NOTE: This function will only work with the predefined load routine defined in 
              calc_spc_temperature_anisotropy.py
    """
    time = get_data(ds, 'epoch')

    peak_az = get_data(ds, 'peak_azimuth')
    peak_el = get_data(ds, 'peak_elevation')

    # Skipping the previous load of the average flow angles...
    vx =  get_data(ds, 'vel_x')          # Defined from vx = vz tan(azimuth)
    vy =  get_data(ds, 'vel_y')          # Defined from vy = vz tan(elevation)
    vz = -get_data(ds, 'vel_mid')        # Negative included to keep SPC Frame convention

    # Now we are going to get the magnetic field data loaded in
    bvec = np.atleast_2d(get_data(ds, 'b_vec'))    # This ensures that the dataset is atleast 2d
                                              # allowing for passing of a single time instence
                                              # or all times in an array.
    nvec = bvec/np.linalg.norm(bvec, axis=1)[:,None]  # NOTE: axis 1 is always the unit vector dimension.

    # NOTE: The previous version of this code seemed to merge all the unit vectors into a 
    #       single array of N_elements. Therefore, for now I will design the function to
    #       work for a single measurement in time, then I will add a group command after.

    vz_bx = vz * nvec[:,0][:,None]   # They share the same time dimension
    vz_by = vz * nvec[:,1][:,None]  

    rdf = get_data(ds, 'rdf')
    check_flux = False
    if 'diff_charge_flux_density' in ds:
        charge_flux = get_data(ds, 'diff_charge_flux_density')
        check_flux = True

    weights = rdf/np.nanmax(rdf, axis=1)[:,None]

    # Only select the points that are above out threshold value
    mask_weights1 = np.isfinite(weights) 
    mask_weights2 = weights > threshold

    mask_weights = mask_weights1 & mask_weights2

    if NO_WEIGHTS:
        mask_weights = np.isfinite(weights)     # We still only want finite values.
        weights = np.ones_like(weights)

    tani_chi = np.zeros(len(time))
    tani_psi = np.zeros(len(time))
    all_results = []
    for i in tqdm(range(len(time))):
        # Case 1: skip if entire slice is useless
        if check_flux and np.all(np.isnan(charge_flux[i])):
            print(f'Charge flux is all NaN. Current too low! Skipping index {i}')
            # append NaN-filled dataset for this timestamp
            all_results.append(
                compute_anisotropy_results(
                    np.nan*np.ones(2), np.nan, np.nan,
                    np.nan*np.ones(2), np.nan, np.nan,
                    np.nan, np.nan, np.nan, uncertainty_map,
                    np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan,
                    time[i], ds, [np.nan, np.nan, np.nan]
                )
            )
            continue

        if np.all(np.isnan(vz_bx[i])) or np.all(np.isnan(vz_by[i])):
            print(f'All NaN slice encountered. Skipping index {i}')
            all_results.append(
                compute_anisotropy_results(
                    np.nan*np.ones(2), np.nan, np.nan,
                    np.nan*np.ones(2), np.nan, np.nan,
                    np.nan, np.nan, np.nan, uncertainty_map,
                    np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan,
                    time[i], ds, [np.nan, np.nan, np.nan]
                )
            )
            continue
        vzbx_fit    = vz_bx[i][mask_weights[i]]
        vzby_fit    = vz_by[i][mask_weights[i]]

        vx_fit      = vx[i][mask_weights[i]]
        vy_fit      = vy[i][mask_weights[i]]

        weight_fits = weights[i][mask_weights[i]]

        # Case 2: not enough points to fit -> NaNs
        if len(vzbx_fit) < 2 or len(vzby_fit) < 2:
            print(f'Not enough points to fit at index {i}')
            all_results.append(
                compute_anisotropy_results(
                    np.nan*np.ones(2), np.nan, np.nan,
                    np.nan*np.ones(2), np.nan, np.nan,
                    np.nan, np.nan, np.nan, uncertainty_map,
                    np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan,
                    time[i], ds, [np.nan, np.nan, np.nan]
                )
            )
            continue

        # if len(vzbx_fit) < 2: 
        #     results_x = np.ones(6) * np.nan
        # else:
        results_x = find_slopes(linear_model, vzbx_fit, vx_fit, 1./weight_fits, 'lm')
        params_chi, cov_chi, chi_linear_fit, r_sq_chi, chi_sq_chi, slope_err_chi = results_x

        # if len(vzby_fit) < 2:
        #     results_y = np.ones(6) * np.nan
        # else:
        results_y = find_slopes(linear_model, vzby_fit, vy_fit, 1./weight_fits, 'lm')
        params_psi, cov_psi, psi_linear_fit, r_sq_psi, chi_sq_psi, slope_err_psi = results_y

        # Get the anisotropy values!
        alpha_chi, Tani_chi = _calc_anisotropy_from_fits(params_chi, nvec[i,2])
        alpha_psi, Tani_psi = _calc_anisotropy_from_fits(params_psi, nvec[i,2])

        # Define the magnetic field flow angles
        xangle = np.degrees(np.arctan(nvec[i, 0] / nvec[i, 2]))
        yangle = np.degrees(np.arctan(nvec[i, 1] / nvec[i, 2]))   
        
        tani_chi[i] = Tani_chi
        tani_psi[i] = Tani_psi

        result_ds = compute_anisotropy_results(params_chi, slope_err_chi, Tani_chi,
                               params_psi, slope_err_psi, Tani_psi,
                               nvec[i,2], xangle, yangle, uncertainty_map,
                               nvec[i,0], nvec[i,1], r_sq_chi, r_sq_psi,
                               chi_sq_chi, chi_sq_psi, peak_az[i],
                               peak_el[i], time[i], ds, bvec[i])

        all_results.append(result_ds)
    
    results = xr.concat(all_results, dim='time')

    return results

def spc_calculate_anisotropy_groups(ds, uncertainty_map, threshold=0.5, NO_WEIGHTS=False, groups=None, WIN_MAX=None, USE_AVG_BFIELD=False):
    """
    Calculate SPC anisotropy either per timestamp or over user-defined groups.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with SPC data.
    uncertainty_map : array-like
        Map of uncertainties for anisotropy calculation.
    threshold : float
        RDF weighting threshold.
    NO_WEIGHTS : bool
        If True, ignore weights (all weights=1).
    groups : list of tuple, optional
        Each element is (start, stop) np.datetime64 or pd.Timestamp.
        If None, compute anisotropy per timestamp.
    """

    time = get_data(ds, "epoch")

    # Data extraction
    peak_az = get_data(ds, "peak_azimuth")
    peak_el = get_data(ds, "peak_elevation")
    vx = get_data(ds, "vel_x")
    vy = get_data(ds, "vel_y")
    vz = -get_data(ds, "vel_mid")
    bvec = np.atleast_2d(get_data(ds, "b_vec"))
    nvec = bvec / np.linalg.norm(bvec, axis=1)[:, None]
    vz_bx = vz * nvec[:, 0][:, None]
    vz_by = vz * nvec[:, 1][:, None]
    rdf = get_data(ds, "rdf")

    check_flux = False
    if "diff_charge_flux_density" in ds:
        charge_flux = get_data(ds, "diff_charge_flux_density")
        check_flux = True

    weights = rdf / np.nanmax(rdf, axis=1)[:, None]
    mask_weights = np.isfinite(weights) & (weights > threshold)
    if NO_WEIGHTS:
        mask_weights = np.isfinite(weights)
        weights = np.ones_like(weights)

    # --- Package into Dataset for slicing ---
    # We change what's packaged based on the B-field averaging method
    
    base_dict = dict(
        vx=(["time", "dim"], vx),
        vy=(["time", "dim"], vy),
        rdf=(["time", "dim"], rdf),
        weights=(["time", "dim"], weights),
        mask=(["time", "dim"], mask_weights),
        peak_az=("time", peak_az),
        peak_el=("time", peak_el),
        bvec=(["time", "vec"], bvec),
    )

    if not USE_AVG_BFIELD:
        # Original method: Pre-calculate vzbx, vzby, and nvec
        nvec = bvec / np.linalg.norm(bvec, axis=1)[:, None]
        vz_bx = vz * nvec[:, 0][:, None]
        vz_by = vz * nvec[:, 1][:, None]
        base_dict['vzbx'] = (["time", "dim"], vz_bx)
        base_dict['vzby'] = (["time", "dim"], vz_by)
        base_dict['nvec'] = (["time", "vec"], nvec)
    else:
        # New method: Pass raw vz to calculate with avg_bfield inside loop
        base_dict['vz'] = (["time", "dim"], vz)

    base = xr.Dataset(base_dict, coords=dict(time=time))

    all_results = []

    # Case 1: groups not provided → one per timestamp
    if groups is None:
        groups = [(t, t) for t in time.values]

    for (t0, t1) in tqdm(groups):
        block = base.sel(time=slice(t0, t1))

        # Drop all the NaN b-field elements
        block_bvec = np.all(np.isnan(block.bvec.data), axis=1)

        # midpoint timestamp
        # block_time = (block.time.values[0] + (block.time.values[-1] - block.time.values[0])/ 2)
        block_time = (t0 + (t1 - t0)/ 2)

        if np.all(block_bvec) == True:
            # nothing in window
            result_ds = compute_anisotropy_results(
                np.nan * np.ones(2), np.nan, np.nan,
                np.nan * np.ones(2), np.nan, np.nan,
                np.nan, np.nan, np.nan, uncertainty_map,
                np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, block_time,ds, [np.nan, np.nan, np.nan],
            )
            all_results.append(result_ds)
            continue
        elif np.any(block_bvec) == True:
            # Drop the bad index values.
            print('Dropping the NaN index')
            block = block.drop_isel(time=block_bvec)
        

        if block.time.size == 0:
            # nothing in window
            result_ds = compute_anisotropy_results(
                np.nan * np.ones(2), np.nan, np.nan,
                np.nan * np.ones(2), np.nan, np.nan,
                np.nan, np.nan, np.nan, uncertainty_map,
                np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, block_time,ds, [np.nan, np.nan, np.nan],
            )
            all_results.append(result_ds)
            continue

        if WIN_MAX:
            weights = block.rdf.values / np.nanmax(block.rdf.values)
            weight_mask = np.isfinite(weights) & (weights > threshold)

            valid_idx = weight_mask
            weight_fits = weights[valid_idx]
        else:
            # Stack all points in block
            valid_idx = block.mask.values
            weight_fits = block.weights.values[valid_idx]

        # Get vx and vy for fitting
        vx_fit = block.vx.values[valid_idx]
        vy_fit = block.vy.values[valid_idx]

        # --- NEW: Get vzbx, vzby, and n_avg based on method ---
        if not USE_AVG_BFIELD:
            # Original Method: Use pre-calculated, stacked values
            vzbx_fit = block.vzbx.values[valid_idx]
            vzby_fit = block.vzby.values[valid_idx]
            
            # Calculate mean n-vector from the stack of n-vectors
            block_nvec_avg = np.nanmean(block.nvec.values, axis=0)
            block_nvec_avg /= np.linalg.norm(block_nvec_avg)
        else:
            # New Method: Use block-averaged B-field
            
            # Calculate mean B-field vector for the block
            block_bvec_avg_calc = np.nanmean(block.bvec.values, axis=0)
            
            # Normalize it to get the single mean N-vector for the block
            block_nvec_avg = block_bvec_avg_calc / np.linalg.norm(block_bvec_avg_calc)
            
            # Get all 'vz' values from the block
            vz_all_in_block = block.vz.values
            
            # Calculate vz_bx and vz_by using the SINGLE average n-vector
            # (vz (time, dim) * nvec_x (scalar))
            vz_bx_block = vz_all_in_block * block_nvec_avg[0] 
            vz_by_block = vz_all_in_block * block_nvec_avg[1] 
            
            # Apply the same mask as all other variables
            vzbx_fit = vz_bx_block[valid_idx]
            vzby_fit = vz_by_block[valid_idx]
        # --- END NEW LOGIC ---
        

        if len(vzbx_fit) < 2 or len(vzby_fit) < 2:
            result_ds = compute_anisotropy_results(
                np.nan * np.ones(2), np.nan, np.nan,
                np.nan * np.ones(2), np.nan, np.nan,
                np.nan, np.nan, np.nan, uncertainty_map,
                np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, block_time,ds, [np.nan, np.nan, np.nan],
            )
            all_results.append(result_ds)
            continue

        # mean b-vector
        block_bvec_to_save = np.nanmean(block.bvec.values, axis=0)

        # Fits
        results_x = find_slopes(linear_model, vzbx_fit, vx_fit, 1.0 / weight_fits, "lm")
        params_chi, cov_chi, chi_linear_fit, r_sq_chi, chi_sq_chi, slope_err_chi = results_x

        results_y = find_slopes(linear_model, vzby_fit, vy_fit, 1.0 / weight_fits, "lm")
        params_psi, cov_psi, psi_linear_fit, r_sq_psi, chi_sq_psi, slope_err_psi = results_y

        alpha_chi, Tani_chi = _calc_anisotropy_from_fits(params_chi, block_nvec_avg[2])
        alpha_psi, Tani_psi = _calc_anisotropy_from_fits(params_psi, block_nvec_avg[2])

        xangle = np.degrees(np.arctan(block_nvec_avg[0] / block_nvec_avg[2]))
        yangle = np.degrees(np.arctan(block_nvec_avg[1] / block_nvec_avg[2]))

        # --- NEW: average L3 values for this block ---
        l3_avgs = {}
        for key in ["np_fit", "np_moment", "wp_fit", "wp_moment", "vp_fit_SC", "vp_moment_SC"]:
            if key in block:
                l3_avgs[key] = float(block[key].mean(skipna=True).values)
            else:
                l3_avgs[key] = np.nan

        result_ds = compute_anisotropy_results(
            params_chi,
            slope_err_chi,
            Tani_chi,
            params_psi,
            slope_err_psi,
            Tani_psi,
            block_nvec_avg[2],
            xangle,
            yangle,
            uncertainty_map,
            block_nvec_avg[0],
            block_nvec_avg[1],
            r_sq_chi,
            r_sq_psi,
            chi_sq_chi,
            chi_sq_psi,
            block.peak_az.mean().item(),
            block.peak_el.mean().item(),
            block_time,
            ds,
            block_bvec_to_save,
        )
        all_results.append(result_ds)

    return xr.concat(all_results, dim="time")
