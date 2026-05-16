"""
This is the final updated version of the spc temperature anisotropy code. 
The previous version of this code was called spc_calc_anisotropy_v3.py, however, 
it become too large with minor changes here and there, that it become untractable. 

For this code we adopt a completely Bayesian approach to the fitting process. 

Author: Michael Terres
"""
import os
import emcee
import corner
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from numba import njit
from tqdm import tqdm

# Import the scipy packages
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde
from scipy.ndimage import label

BASE_NAN_TEMPLATE = {
    "Nx": np.nan, "Ny": np.nan, "Nz": np.nan,
    "Bx": np.nan, "By": np.nan, "Bz": np.nan,
    "B_x_flow_angle": np.nan, "B_y_flow_angle": np.nan,
    "peak_azimuth": np.nan, "peak_elevation": np.nan,
    
    "T_ani_avg": np.nan, "T_ani_avg_uncert": np.nan, 
    "T_ani_avg_delta_plus": np.nan, "T_ani_avg_delta_minus": np.nan,
    
    "T_ani_chi": np.nan, "T_ani_chi_delta_plus": np.nan, "T_ani_chi_delta_minus": np.nan,
    "T_ani_psi": np.nan, "T_ani_psi_delta_plus": np.nan, "T_ani_psi_delta_minus": np.nan,
    
    "slope_chi": np.nan, "r_sq_chi": np.nan, "chi_sq_chi": np.nan,
    "slope_psi": np.nan, "r_sq_psi": np.nan, "chi_sq_psi": np.nan,
    
    "chi_sq_joint": np.nan, "chi_sq_x": np.nan, "chi_sq_y": np.nan,
}

def get_data(obj, key):
    """
    Helper function that delineates between dictonary and xarray. 

    Inputs:
    -------
    obj : xr.dataset or dictonary
        Input dataset
    key : string
        Key that needs to be extracted
    """
    val = obj[key]
    if isinstance(val, xr.DataArray):
        return val.data
    return val

def get_contiguous_peak_mask(weights, threshold):
    """
    Takes a 2D array of VDF weights (time, velocity) and returns a boolean mask
    keeping only the contiguous points around the peak of each row.
    """
    # Create the raw threshold mask
    raw_mask = np.isfinite(weights) & (weights > threshold)
    
    # The Magic Matrix: 
    # This 3x3 array tells scipy to only group pixels horizontally (velocity bins)
    # and NEVER group pixels vertically (adjacent time steps).
    structure = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]])
    
    # Label the islands
    labeled_array, _ = label(raw_mask, structure=structure)
    
    # Find the exact index of the peak for each row
    peak_indices = np.zeros(weights.shape[0], dtype=int)
    valid_rows = np.any(np.isfinite(weights), axis=1) # Protect against all-NaN rows
    peak_indices[valid_rows] = np.nanargmax(weights[valid_rows], axis=1)
    
    # Extract the island label that belongs to the peak
    row_indices = np.arange(weights.shape[0])
    peak_labels = labeled_array[row_indices, peak_indices]
    
    # Build the clean mask: match the peak's label and ignore 0 (background)
    clean_mask = raw_mask & (labeled_array == peak_labels[:, None]) & (labeled_array > 0)
    
    return clean_mask

def package_to_xarray(data_vars, timestamp, epoch_start, epoch_end):
    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": [timestamp]},
        attrs={"Note": f"Fitting Results for {epoch_start} to {epoch_end}"}
    ) 

@njit
def _calculate_empirical_guess(vzb, v_trans, var_array, bz):
    """
    Calculates the inverse-variance weighted OLS slope and converts 
    it to an initial guess for the true anisotropy.
    """
    W = 1.0 / var_array
    sum_W = np.sum(W)
    
    # Catch empty or completely invalid arrays
    if sum_W <= 0 or np.isnan(sum_W):
        return np.nan, 0.0
        
    c_vzb = np.sum(W * vzb) / sum_W
    c_vtrans = np.sum(W * v_trans) / sum_W
    
    variance_vzb = np.sum(W * (vzb - c_vzb)**2)
    
    # Protect against perfectly vertical data (division by zero)
    if variance_vzb < 1e-12:
        return np.nan, 0.0
        
    slope = np.sum(W * (vzb - c_vzb) * (v_trans - c_vtrans)) / variance_vzb
    
    # Convert slope to anisotropy
    denom = slope * (1.0 - bz**2) + bz
    if np.abs(denom) < 1e-12:
        return np.nan, sum_W
        
    emp_alpha = (-slope / denom) + 1.0
    
    return emp_alpha, sum_W

@njit
def log_likelihood_hierarchical_3d(theta, vzbx, vx, vzby, vy, bz, var_x_array, var_y_array):
    """
    This is the log-likelihood function 
    """
    log_alpha_true, log_alpha_eff_x, log_alpha_eff_y = theta

    alpha_eff_x = 10**(log_alpha_eff_x)
    alpha_eff_y = 10**(log_alpha_eff_y)

    # Map Proposed Anisotorpy to Physical Slopes
    expected_slope_x = ((1.0 - alpha_eff_x) * bz) / (1.0 - (1.0 - alpha_eff_x) * (1.0 - bz**2))
    expected_slope_y = ((1.0 - alpha_eff_y) * bz) / (1.0 - (1.0 - alpha_eff_y) * (1.0 - bz**2))
    
    # Dynamically Calculate Optimal Intercepts (Inverse Variance Weighted)
    W_x = 1.0 / var_x_array
    W_y = 1.0 / var_y_array
    
    intercept_x = np.sum(W_x * (vx - expected_slope_x * vzbx)) / np.sum(W_x)
    intercept_y = np.sum(W_y * (vy - expected_slope_y * vzby)) / np.sum(W_y)
    
    # Build the Theoretical Model Line
    mu_x = expected_slope_x * vzbx + intercept_x
    mu_y = expected_slope_y * vzby + intercept_y
    
    # Calculate the Gaussian Log-Likelihood Penalty
    # Note: 1.837877 is ln(2*pi). Since var_array is pre-calculated and constant, 
    # it technically drops out of the relative MCMC math, but it's formally correct to keep it!
    L_x = np.sum(((vx - mu_x)**2 / var_x_array) + 1.837877 + np.log(var_x_array))
    L_y = np.sum(((vy - mu_y)**2 / var_y_array) + 1.837877 + np.log(var_y_array))
    
    return -0.5 * (L_x + L_y)    

# @njit
# def log_prior_hierarchical_3d(theta, ani_list, dp_slice_x, dm_slice_x, bias_slice_x,
#                                                dp_slice_y, dm_slice_y, bias_slice_y):
#     log_alpha_true, log_alpha_eff_x, log_alpha_eff_y = theta

#     if not (-2.0 < log_alpha_true  < 2.0): return -np.inf
#     if not (-2.0 < log_alpha_eff_x < 2.0): return -np.inf
#     if not (-2.0 < log_alpha_eff_y < 2.0): return -np.inf

#     alpha_true = 10**(log_alpha_true)
    
#     # Make sure we do not exceed the boundaries.
#     alpha_true_lookup = max(min(alpha_true, ani_list[-1]), ani_list[0]) #np.clip(alpha_true, ani_list[0], ani_list[-1])

#     ### Calculate the log prior for x
#     valid_x = np.isfinite(dp_slice_x) & np.isfinite(dm_slice_x) & np.isfinite(bias_slice_x)

#     if np.sum(valid_x) < 2:
#         log_P_eff_x = 0.0
#     else:
#         log_dp_x = np.interp(alpha_true_lookup, ani_list[valid_x], dp_slice_x[valid_x])
#         log_dm_x = np.interp(alpha_true_lookup, ani_list[valid_x], dm_slice_x[valid_x])
#         log_bias_x = np.interp(alpha_true_lookup, ani_list[valid_x], bias_slice_x[valid_x])

#         expected_x = log_alpha_true + log_bias_x

#         sig_x = log_dp_x if log_alpha_eff_x > expected_x else log_dm_x
#         sig_x = max(sig_x, 1e-3)
#         # Probability calculation
#         log_P_eff_x = -np.log(log_dp_x + log_dm_x) - 0.5 * ((log_alpha_eff_x - expected_x) / sig_x)**2

#     ### Calculate the log prior for y
#     valid_y = np.isfinite(dp_slice_y) & np.isfinite(dm_slice_y) & np.isfinite(bias_slice_y)

#     if np.sum(valid_y) < 2:
#         log_P_eff_y = 0.0
#     else:
#         log_dp_y = np.interp(alpha_true, ani_list[valid_y], dp_slice_y[valid_y])
#         log_dm_y = np.interp(alpha_true, ani_list[valid_y], dm_slice_y[valid_y])
#         log_bias_y = np.interp(alpha_true, ani_list[valid_y], bias_slice_y[valid_y])
    
#         expected_y = log_alpha_true + log_bias_y
#         sig_y = log_dp_y if log_alpha_eff_y > expected_y else log_dm_y
#         sig_y = max(sig_y, 1e-3)
#         # Probability calculation
#         log_P_eff_y = -np.log(log_dp_y + log_dm_y) - 0.5 * ((log_alpha_eff_y - expected_y) / sig_y)**2
        
#     return log_P_eff_x + log_P_eff_y

@njit
def log_prior_hierarchical_3d(theta, ani_list, dp_slice_x, dm_slice_x, bias_slice_x,
                                               dp_slice_y, dm_slice_y, bias_slice_y):
    log_alpha_true, log_alpha_eff_x, log_alpha_eff_y = theta

    if not (-2.0 < log_alpha_true  < 2.0): return -np.inf
    if not (-2.0 < log_alpha_eff_x < 2.0): return -np.inf
    if not (-2.0 < log_alpha_eff_y < 2.0): return -np.inf

    log_ani_list = np.log10(ani_list)
    
    # --- THE FIX: Evaluate the map based on the measured effective values! ---
    log_eff_x_lookup = max(min(log_alpha_eff_x, log_ani_list[-1]), log_ani_list[0])
    log_eff_y_lookup = max(min(log_alpha_eff_y, log_ani_list[-1]), log_ani_list[0])

    ### Calculate the log prior for x
    valid_x = np.isfinite(dp_slice_x) & np.isfinite(dm_slice_x)

    if np.sum(valid_x) < 2:
        log_P_eff_x = 0.0
    else:
        # Interpolate using log_eff_x_lookup
        log_dp_x = np.interp(log_eff_x_lookup, log_ani_list[valid_x], dp_slice_x[valid_x])
        log_dm_x = np.interp(log_eff_x_lookup, log_ani_list[valid_x], dm_slice_x[valid_x])

        expected_x = log_alpha_true 
        sig_x = log_dp_x if log_alpha_eff_x > expected_x else log_dm_x
        sig_x = max(sig_x, 1e-3)
        log_P_eff_x = -np.log(log_dp_x + log_dm_x) - 0.5 * ((log_alpha_eff_x - expected_x) / sig_x)**2

    ### Calculate the log prior for y
    valid_y = np.isfinite(dp_slice_y) & np.isfinite(dm_slice_y) 

    if np.sum(valid_y) < 2:
        log_P_eff_y = 0.0
    else:
        # Interpolate using log_eff_y_lookup
        log_dp_y = np.interp(log_eff_y_lookup, log_ani_list[valid_y], dp_slice_y[valid_y])
        log_dm_y = np.interp(log_eff_y_lookup, log_ani_list[valid_y], dm_slice_y[valid_y])
        
        expected_y = log_alpha_true 
        sig_y = log_dp_y if log_alpha_eff_y > expected_y else log_dm_y
        sig_y = max(sig_y, 1e-3)
        log_P_eff_y = -np.log(log_dp_y + log_dm_y) - 0.5 * ((log_alpha_eff_y - expected_y) / sig_y)**2
        
    return log_P_eff_x + log_P_eff_y

@njit
def log_probability_hierarchical_3d(theta, vzbx, vx, vzby, vy, bz, ani_list, 
                                    dp_slice_x, dm_slice_x, bias_slice_x,
                                    dp_slice_y, dm_slice_y, bias_slice_y,
                                    var_x_array, var_y_array):
    # Define the log prior
    lp = log_prior_hierarchical_3d(theta, ani_list, dp_slice_x, dm_slice_x, bias_slice_x,
                                                          dp_slice_y, dm_slice_y, bias_slice_y)
    
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood_hierarchical_3d(theta, vzbx, vx, vzby, vy, bz, var_x_array, var_y_array)

    if np.isnan(ll):
        return -np.inf

    return lp + ll

def run_emcee_hierarchical_3d(vzbx, vx, vzby, vy, bz, uncert_map, 
                              xangle, yangle, var_x_array, var_y_array):
    """
    This function is the nuts and bolts for the 3D hierarchical Bayesian
    fitting. As the function name suggests we are using the EMCEE package
    to sample our defined parameter space. 
    """
    
    # Here we are going to guess our starting point using the linear fit 
    # to the x and y data. 
    emp_alpha_x, weight_x = _calculate_empirical_guess(vzbx, vx, var_x_array, bz)
    emp_alpha_y, weight_y = _calculate_empirical_guess(vzby, vy, var_y_array, bz)

    # Safely combine them using their statistical weights
    alphas = np.array([emp_alpha_x, emp_alpha_y])
    weights = np.array([weight_x, weight_y])
    
    valid = np.isfinite(alphas) & (weights > 0)
    
    if np.any(valid):
        # Weighted average naturally favors the plane with the cleaner signal
        emp_alpha = np.average(alphas[valid], weights=weights[valid])
    else:
        # Ultimate fallback if both planes fail
        emp_alpha = 1.0

    # Add a buffer so that it does not fail imediately. 
    emp_alpha = np.clip(emp_alpha, 0.15, 9.85)
    log_emp_alpha = np.log10(emp_alpha)

    try:
        angle_list, ani_list, dp_map, dm_map, bias_map = uncert_map
    except ValueError:
        angle_list, ani_list, dp_map, dm_map = uncert_map
        bias_map = np.zeros_like(dp_map)

    # Note the maps are (Anistropy, Theta). 

    # Get the nearest angular slice. 
    idx_x = np.nanargmin(np.abs(np.abs(xangle) - angle_list))
    idx_y = np.nanargmin(np.abs(np.abs(yangle) - angle_list))

    # Choose the correct x and y slices for dp, dm, and bias
    dp_slice_x = dp_map[:, idx_x]
    dm_slice_x = dm_map[:, idx_x]
    bias_slice_x = bias_map[:, idx_x]

    dp_slice_y = dp_map[:, idx_y]
    dm_slice_y = dm_map[:, idx_y]
    bias_slice_y = bias_map[:, idx_y]

    # Initialize the walkers
    nwalkers = 32
    ndim = 3        # This is the 3D code.

    # Spawn a cloud of random walkers near the empirical alpha
    log_pos_alpha_true  = log_emp_alpha + 1e-2 * np.random.randn(nwalkers)
    log_pos_alpha_eff_x = log_emp_alpha + 1e-2 * np.random.randn(nwalkers)
    log_pos_alpha_eff_y = log_emp_alpha + 1e-2 * np.random.randn(nwalkers)

    # Define the prior space. 
    pos = np.vstack((log_pos_alpha_true, log_pos_alpha_eff_x, log_pos_alpha_eff_y)).T
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_hierarchical_3d,
                                    args = (vzbx, vx, vzby, vy, bz, ani_list, 
                                    dp_slice_x, dm_slice_x, bias_slice_x, 
                                    dp_slice_y, dm_slice_y, bias_slice_y, 
                                    var_x_array, var_y_array)
            )
    
    sampler.run_mcmc(pos, 250, progress=False)

    log_samples = sampler.get_chain(discard=100, flat=True)
    samples = 10**(log_samples)

    samples_true = samples[:,0]

    alpha_16, alpha_50, alpha_84 = np.percentile(samples_true, [16, 50, 84])
    sig_minus = alpha_50 - alpha_16
    sig_plus = alpha_84 - alpha_50

    return alpha_50, sig_plus, sig_minus, samples

def _run_bayesian_pipeline(block, valid_idx, vzbx_fit, vx_fit, 
                           vzby_fit, vy_fit, vz_var, block_nvec_avg,
                           uncertainty_map, xangle, yangle, static_length):

    sig_I = 1.0 
    lamda = 1.009       # Assuming 1 pA noise floor, and lamda is a defined constant
    C = lamda/2.0

    # Grab in the plate currents that will be used for defining the plate differences!
    ia_fit = block.ia_curr.values[valid_idx]
    ib_fit = block.ib_curr.values[valid_idx]
    ic_fit = block.ic_curr.values[valid_idx]
    id_fit = block.id_curr.values[valid_idx]

    # Grab the total current. 
    i_tot = np.maximum(ia_fit + ib_fit + ic_fit + id_fit, 1e-6)     # Make sure we do not divide by zero.
    i_diff_x = (ia_fit + id_fit) - (ib_fit + ic_fit)
    i_diff_y = (ic_fit + id_fit) - (ia_fit + ib_fit)

    vz_fit = block.vz.values[valid_idx] 
    vz_fit_abs = np.abs(vz_fit)

    # Derive the Velocity Window Variance. Appendix C!
    theta_x_rad = np.arctan(vx_fit / vz_fit_abs)    # Used in current variance
    theta_y_rad = np.arctan(vy_fit / vz_fit_abs)

    # Recall the voltage window partial derivative
    # partial Delta_k / partial v_z = tan(theta) - M bx = vx/vz - M b_x 
    # Derive an empircal slope for the M b_k term.
    # NOTE: We use the oridinary least squares to fix our uncertainty!
    cov_x = np.cov(vzbx_fit, vx_fit)    
    emp_slope_x = cov_x[0, 1] / max(cov_x[0, 0], 1e-6)
    cov_y = np.cov(vzby_fit, vy_fit)
    emp_slope_y = cov_y[0, 1] / max(cov_y[0, 0], 1e-6)

    # Get the unit vector values
    nx, ny = block_nvec_avg[0], block_nvec_avg[1]   

    # Evalute the voltage window variance. Both x and y.
    deriv_x = (vx_fit / vz_fit) - (emp_slope_x * nx)
    deriv_y = (vy_fit / vz_fit) - (emp_slope_y * ny)

    # Propogate through to get the variance term for voltage window. Appendix C1
    term1_x = vz_var * (deriv_x**2)
    term1_y = vz_var * (deriv_y**2)
    
    # Recall the current partial derivative
    # partial Delta_k/partial phi = vz sec^2 phi.
    # Define the sec^4 term
    geom_mod_x = 1.0 / (np.cos(theta_x_rad)**4)
    geom_mod_y = 1.0 / (np.cos(theta_y_rad)**4)

    # When we convert from sigma_phi/theta to sigma_I introduce the
    # term: (C^2 / I_tot^2) [1 + (I_diff/I_tot)**2]
    prefix = (4.0*(sig_I**2) * (vz_fit_abs**2) * (C**2)) / (i_tot**2)

    # Define the second variance term for the currents.
    term2_x = prefix * geom_mod_x * (1.0 + (i_diff_x / i_tot)**2)
    term2_y = prefix * geom_mod_y * (1.0 + (i_diff_y / i_tot)**2)

    # Total variance that is being passed into the model. 
    var_x_array = np.maximum(term1_x + term2_x, 1e-4)
    var_y_array = np.maximum(term1_y + term2_y, 1e-4)

    # TODO: We should consider adding in the magnetic field variance terms here

    avg_alpha, avg_sig_plus, avg_sig_minus, samples = run_emcee_hierarchical_3d(
        vzbx_fit, vx_fit, vzby_fit, vy_fit, block_nvec_avg[2], 
        uncertainty_map, xangle, yangle, var_x_array, var_y_array
    )

    # JOINT REDUCED CHI-SQUARED: This tells us how good out fit is.
    alpha_eff_x = np.median(samples[:, 1])
    alpha_eff_y = np.median(samples[:, 2])
    bz = block_nvec_avg[2]
    
    # Get the effective x and y fits.
    m_eff_x = ((1.0 - alpha_eff_x) * bz) / (1.0 - (1.0 - alpha_eff_x) * (1.0 - bz**2))
    m_eff_y = ((1.0 - alpha_eff_y) * bz) / (1.0 - (1.0 - alpha_eff_y) * (1.0 - bz**2))
    
    W_x = 1.0 / var_x_array
    W_y = 1.0 / var_y_array
    b_eff_x = np.sum(W_x * (vx_fit - m_eff_x * vzbx_fit)) / np.sum(W_x)
    b_eff_y = np.sum(W_y * (vy_fit - m_eff_y * vzby_fit)) / np.sum(W_y)
    
    sq_res_x_valid = ((vx_fit - (m_eff_x * vzbx_fit + b_eff_x))**2) / var_x_array
    sq_res_y_valid = ((vy_fit - (m_eff_y * vzby_fit + b_eff_y))**2) / var_y_array
    
    # norm_res_sq_x = np.full(static_length, np.nan)
    # norm_res_sq_y = np.full(static_length, np.nan)
    
    # n_valid = len(sq_res_x_valid)
    # if n_valid > 0:
    #     fill_len = min(n_valid, static_length) 
    #     norm_res_sq_x[:fill_len] = sq_res_x_valid[:fill_len]
    #     norm_res_sq_y[:fill_len] = sq_res_y_valid[:fill_len]

    # Create an emply 2D array shaped exactly like the input block (time, dim)
    block_shape = valid_idx.shape
    block_res_x = np.full(block_shape, np.nan)
    block_res_y = np.full(block_shape, np.nan)
    # Map the 1D valid residuals back into their exact physical velocity bins
    block_res_x[valid_idx] = sq_res_x_valid
    block_res_y[valid_idx] = sq_res_y_valid 

    # Just to keep the terminal from exploding with warnings. 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        norm_res_sq_x = np.nanmean(block_res_x, axis=0)
        norm_res_sq_y = np.nanmean(block_res_y, axis=0)
    
    # Degrees of freedom, used for normalizing the Chi-squared values.
    dof_single = max(len(vzbx_fit) - 2.0, 1.0)
    dof_joint = max((2.0 * len(vzbx_fit)) - 3.0, 1.0)
    
    chi_sq_x = np.sum(sq_res_x_valid) / dof_single
    chi_sq_y = np.sum(sq_res_y_valid) / dof_single
    chi_sq_joint = (np.sum(sq_res_x_valid) + np.sum(sq_res_y_valid)) / dof_joint

    update_dict = {
        "T_ani_avg": avg_alpha,
        "T_ani_avg_delta_plus": avg_sig_plus,
        "T_ani_avg_delta_minus": avg_sig_minus,
        "chi_sq_joint": chi_sq_joint,
        "chi_sq_x": chi_sq_x,
        "chi_sq_y": chi_sq_y,
        "norm_res_sq_x": (["dim"], norm_res_sq_x),
        "norm_res_sq_y": (["dim"], norm_res_sq_y),
    }

    # We also return samples and var arrays so the main loop can handle plotting!
    return update_dict, samples, var_x_array, var_y_array



### This is the main code block that we are using!
def calculate_spc_anisotropy(ds, uncertainty_map, method='Bayesian_MCMC',
                              threshold=0.5, no_weights=False, groups=None,
                              win_max=False, use_avg_bfield=False, plot_diagnostics=False):
    """
    Unified pipeline for calculating SPC core proton temperature anisotropy.
    Handles both single-timestamp events (if groups=None) and grouped intervals.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with SPC data.
    uncertainty_map : tuple
        Map of uncertainties (angle_list, ani_list, delta_plus, delta_minus).
    method : str
        'Linear' or 'Bayesian_MCMC'
    threshold : float
        RDF weighting threshold.
    no_weights : bool
        If True, ignore weights (all weights=1).
    groups : list of tuple, optional
        Each element is (start, stop) np.datetime64 or pd.Timestamp.
        If None, compute anisotropy per individual timestamp.
    win_max : bool
        If True, only use the maximum weight in each velocity bin across the time window.
    use_avg_bfield : bool
        If True, calculate the projections using the average B-field of the block.
    plot_diagnostics : bool
        If True, saves MCMC corner and diagnostic plots.
    """
    print(f"Running SPC Anisotropy Pipeline | Method = {method} | Groups = {"None (Single Events)" if groups is None else "Provided"}")

    # From the input dataset, extract all the key supporting information.
    time = get_data(ds, "epoch")

    peak_az = get_data(ds, "peak_azimuth")      # Peak flow angle values
    peak_el = get_data(ds, "peak_elevation")    
    vx = get_data(ds, "vel_x")                  # Derived from vx = vz tan(az)
    vy = get_data(ds, "vel_y")                  # Derived from vx = vz tan(el)
    vz = -get_data(ds, "vel_mid")               # Add the negative to rectify sign
    vz_hi = -get_data(ds, "vel_hi")             
    vz_lo = -get_data(ds, "vel_lo")
    bvec = np.atleast_2d(get_data(ds, "b_vec")) 

    # Extract the plate currents and the reduced distribution function
    ia_curr = get_data(ds, "a_current")
    ib_curr = get_data(ds, "b_current")
    ic_curr = get_data(ds, "c_current")
    id_curr = get_data(ds, "d_current")

    # Get the reduced distribution function
    rdf = get_data(ds, "rdf")

    # Define the weights
    weights = rdf/np.nanmax(rdf, axis=1)[:,None]    # Normalized each VDF to 1, then we will take a thermal width!

    if no_weights:      # This is a very rare flag to pass and is generally not used.
        mask_weights = np.isfinite(weights)
        weights = np.ones_like(weights)
    else:
        mask_weights = get_contiguous_peak_mask(weights, threshold)

    # Package into Dataset for easy block slicing
    base_dict = dict(
        vx=(["time", "dim"], vx),
        vy=(["time", "dim"], vy),
        vz=(["time", "dim"], vz),
        rdf=(["time", "dim"], rdf),
        weights=(["time", "dim"], weights),
        mask=(["time", "dim"], mask_weights),
        peak_az=("time", peak_az),
        peak_el=("time", peak_el),
        bvec=(["time", "vec"], bvec),
        ia_curr=(["time", "dim"], ia_curr),
        ib_curr=(["time", "dim"], ib_curr),
        ic_curr=(["time", "dim"], ic_curr),
        id_curr=(["time", "dim"], id_curr),
    )

    # Pre-calculate the magnetic field geometry if not averaging over the block
    if not use_avg_bfield:
        nvec = bvec / np.linalg.norm(bvec, axis=1)[:,None]
        base_dict['vzbx'] = (["time", "dim"], vz * nvec[:, 0][:, None])
        base_dict['vzby'] = (["time", "dim"], vz * nvec[:, 1][:, None])
        base_dict['nvec'] = (["time", "vec"], nvec)

    # Vz Error estimation, this comes in for uncertainty calculations
    vz_geo_center = (vz_hi + vz_lo)/ 2.0
    vz_bias_sq = (vz - vz_geo_center)**2
    base_dict['vz_var'] = (["time", "dim"], ((vz_hi - vz_lo)**2) / 12 + vz_bias_sq)

    base = xr.Dataset(base_dict, coords = dict(time=time))
    static_length = base.vx.shape[1]

    nan_template = BASE_NAN_TEMPLATE.copy()
    nan_template["norm_res_sq_x"] = (["dim"], np.full(static_length, np.nan))
    nan_template["norm_res_sq_y"] = (["dim"], np.full(static_length, np.nan))

    all_results = []

    # Handle Single Events
    if groups is None:
        groups = [(t,t) for t in time.values]

    count = 0
    for (t0, t1) in tqdm(groups):
        count += 1

        block = base.sel(time=slice(t0, t1))
        block_time = t0 + (t1 - t0)/2 

        # Filter out the NaN B-fields
        block_bvec_nan = np.all(np.isnan(block.bvec.data), axis=1)
        if np.all(block_bvec_nan) or block.time.size == 0:
            result_ds = package_to_xarray(nan_template.copy(), block_time, t0, t1)
            all_results.append(result_ds)
            continue
        elif np.any(block_bvec_nan):
            block = block.isel(time=~block_bvec_nan)

        # Windowing and weighting masks
        if win_max:
            weights_block = block.rdf.values / np.nanmax(block.rdf.values)
            valid_idx = get_contiguous_peak_mask(weights_block, threshold)
            weight_fits = weights_block[valid_idx]
        else:
            valid_idx = block.mask.values
            weight_fits = block.weights.values[valid_idx]

        vx_fit = block.vx.values[valid_idx]
        vy_fit = block.vy.values[valid_idx]
        vz_var = block.vz_var.values[valid_idx]

        # Apply Magnetic Field Geometry
        if not use_avg_bfield:
            vzbx_fit = block.vzbx.values[valid_idx]
            vzby_fit = block.vzby.values[valid_idx]
            block_nvec_avg = np.nanmean(block.nvec.values, axis=0)
            block_nvec_avg /= np.linalg.norm(block_nvec_avg)
        else:
            block_bvec_avg_calc = np.nanmean(block.bvec.values, axis=0)
            block_nvec_avg = block_bvec_avg_calc / np.linalg.norm(block_bvec_avg_calc)
            vz_all_in_block = block.vz.values
            vzbx_fit = (vz_all_in_block * block_nvec_avg[0])[valid_idx]
            vzby_fit = (vz_all_in_block * block_nvec_avg[1])[valid_idx]

        if len(vzbx_fit) < 2 or len(vzby_fit) < 2:
            result_ds = package_to_xarray(nan_template.copy(), block_time, t0, t1)
            all_results.append(result_ds)
            continue

        block_bvec_to_save = np.nanmean(block.bvec.values, axis=0)
        xangle = np.degrees(np.arctan(block_nvec_avg[0] / block_nvec_avg[2]))
        yangle = np.degrees(np.arctan(block_nvec_avg[1] / block_nvec_avg[2]))

        data_vars = nan_template.copy()
        data_vars.update({
            "Nx": block_nvec_avg[0],
            "Ny": block_nvec_avg[1],
            "Nz": block_nvec_avg[2],
            "Bx": block_bvec_to_save[0],
            "By": block_bvec_to_save[1],
            "Bz": block_bvec_to_save[2],
            "B_x_flow_angle": xangle,
            "B_y_flow_angle": yangle,
            "peak_azimuth": block.peak_az.mean().item(),
            "peak_elevation": block.peak_el.mean().item(),
        })

        if method == "Bayesian_MCMC":
            update_dict, samples, var_x_array, var_y_array = _run_bayesian_pipeline(
                block, valid_idx, vzbx_fit, vx_fit, vzby_fit, vy_fit, vz_var, 
                block_nvec_avg, uncertainty_map, xangle, yangle, static_length
            )

            data_vars.update(update_dict)

            if plot_diagnostics:
                try:
                    date_folder = str(block_time)[:10] 
                    plot_dir = f'posterior_plots/{date_folder}'
                    os.makedirs(plot_dir, exist_ok=True)        # Make the directory if it does not exist.
                    safe_time = str(block_time).replace(':', '').replace('-', '').replace('T', '_')
                    # Create a corner plot to visualize the 3D parameter space correlations
                    labels = [r"$\alpha_{true}$", r"$\alpha_{eff,x}$", r"$\alpha_{eff,y}$"]
                    fig = corner.corner(np.log10(samples), labels=labels, 
                                        quantiles=[0.16, 0.5, 0.84], show_titles=True)
                    plt.suptitle(f"MCMC Posterior: {block_time}")
                    plt.savefig(f'{plot_dir}/updated_corner_{safe_time}.png', dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Diagnostics plotting failed for {block_time}: {e}")
        else:
            print(f"Method '{method}' is not implemented.")
            pass

        
        result_ds = package_to_xarray(data_vars, block_time, t0, t1)
        all_results.append(result_ds)

    if len(all_results) > 0:
        final_ds = xr.concat(all_results, dim="time")
    else:
        print("Warning: No valid data blocks found. Returning empty Dataset.")
        final_ds = xr.Dataset() 
        
    return final_ds    
            
