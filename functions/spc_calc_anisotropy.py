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
import pandas as pd

from numba import njit
from tqdm import tqdm

from functions.supporting_plots import plot_spc_diagnostic_with_plates

# Import the scipy packages
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde
from scipy.ndimage import label
from scipy.special import ellipe        # Complete elliptic integral of the second kind!

BASE_NAN_TEMPLATE = {
    "Nx": np.nan, "Ny": np.nan, "Nz": np.nan,
    "Bx": np.nan, "By": np.nan, "Bz": np.nan,
    "B_var_x": np.nan, "B_var_y": np.nan, "B_var_z": np.nan, "B_var_tot": np.nan,
    "B_x_flow_angle": np.nan, "B_y_flow_angle": np.nan,
    "peak_azimuth": np.nan, "peak_elevation": np.nan,
    
    # Joint Anisotropy (Linear & Log)
    "T_ani_avg": np.nan,
    "T_ani_avg_delta_plus": np.nan, "T_ani_avg_delta_minus": np.nan,
    "T_ani_avg_delta_plus_log": np.nan, "T_ani_avg_delta_minus_log": np.nan,
    
    # Plane-Specific Effective Anisotropies
    "T_ani_chi": np.nan,
    "T_ani_chi_delta_plus": np.nan, "T_ani_chi_delta_minus": np.nan,
    "T_ani_chi_delta_plus_log": np.nan, "T_ani_chi_delta_minus_log": np.nan,

    "T_ani_psi": np.nan,
    "T_ani_psi_delta_plus": np.nan, "T_ani_psi_delta_minus": np.nan,
    "T_ani_psi_delta_plus_log": np.nan, "T_ani_psi_delta_minus_log": np.nan,
    
    # Intercepts
    "gamma_chi": np.nan, "gamma_chi_delta_plus": np.nan, "gamma_chi_delta_minus": np.nan,
    "gamma_psi": np.nan, "gamma_psi_delta_plus": np.nan, "gamma_psi_delta_minus": np.nan, 

    # Goodness of Fit Statistics
    "chi_sq_joint": np.nan, "chi_sq_chi": np.nan, "chi_sq_psi": np.nan,
    "chi_sq_eff_x": np.nan, "chi_sq_eff_y": np.nan,
    "chi_sq_joint_robust": np.nan, "chi_sq_chi_robust": np.nan, "chi_sq_psi_robust": np.nan,

    "n_samples": np.nan,
}

def calc_vz_hardware_variance(V_lo, V_hi, sig_V_lo, sig_V_hi):
    """
    Computes the hardware variance of vz using a central finite difference 
    of the complete elliptic integral of the second kind.
    """
    # Proton charge-to-mass ratio in C/kg
    q_m = 9.578833e7 
    
    # Define the exact vz function as a nested lambda for easy FD
    # Note: scipy's ellipe takes the parameter m = k^2. 
    # Here, m = (V_hi - V_lo) / V_hi = 1 - (V_lo / V_hi)
    def vz_func(vlo, vhi):
        m = 1.0 - (vlo / vhi)
        # Protect against non-physical negative m or zero V_hi
        m = np.clip(m, 0.0, 1.0)
        return (2.0 / np.pi) * np.sqrt(2.0 * q_m * vhi) * ellipe(m)
    
    # Small delta for numerical derivative (e.g., 0.1 Volts)
    dV = 0.1
    
    # Central finite difference for V_lo
    dvz_dVlo = (vz_func(V_lo + dV, V_hi) - vz_func(V_lo - dV, V_hi)) / (2.0 * dV)
    
    # Central finite difference for V_hi
    dvz_dVhi = (vz_func(V_lo, V_hi + dV) - vz_func(V_lo, V_hi - dV)) / (2.0 * dV)
    
    # Linear error propagation
    # Note: We convert from m/s to km/s by dividing the final variance by 1e6
    var_hw_ms = (dvz_dVlo**2 * sig_V_lo**2) + (dvz_dVhi**2 * sig_V_hi**2)
    
    return var_hw_ms / 1e6

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

@njit(fastmath=True, cache=True, nogil=True)
def _calculate_empirical_guess(vzb, v_trans, var_array, bz):
    """
    Calculates the inverse-variance weighted OLS slope and intercept, 
    and converts the slope to an initial guess for the true anisotropy.
    """
    W = 1.0 / var_array
    sum_W = np.sum(W)
    
    # Catch empty or completely invalid arrays
    if sum_W <= 0 or np.isnan(sum_W):
        return np.nan, np.nan, np.nan, 0.0  # alpha, slope, intercept, sum_W
        
    c_vzb = np.sum(W * vzb) / sum_W
    c_vtrans = np.sum(W * v_trans) / sum_W
    
    variance_vzb = np.sum(W * (vzb - c_vzb)**2)
    
    # Protect against perfectly vertical data (division by zero)
    if variance_vzb < 1e-12:
        return np.nan, np.nan, np.nan, 0.0  # alpha, slope, intercept, sum_W
        
    slope = np.sum(W * (vzb - c_vzb) * (v_trans - c_vtrans)) / variance_vzb
    
    # Calculate the intercept
    intercept = c_vtrans - (slope * c_vzb)
    
    # Convert slope to anisotropy
    denom = slope * (1.0 - bz**2) + bz
    if np.abs(denom) < 1e-12:
        # If the denominator fails, alpha is NaN, but slope and intercept are still mathematically valid!
        return np.nan, slope, intercept, sum_W
        
    emp_alpha = (-slope / denom) + 1.0
    
    return emp_alpha, slope, intercept, sum_W

### Define the 5D Non-Centered Parameterization (NCP) method.
# Theoretically what we are doing is we sample an alpha_*.
# Then, we argue that alpha_eff_k = (alpha_* + bias(alpha_*) + eta_k * sig_k(alpha_*).
# 
# This means we sample the parameter space of Theta = {alpha_*, eta_x, eta_y}.
# The reason we need to do this is we are trying to simulate P(alpha_eff_k | alpha_*)
# however, the measurements have no notion of the instrumental effects, meaning that 
# the MCMC method will attempt to cheat and move to a paramter space that has a large
# uncertainty, rendering the results to be wildly inconsistent if one of the two 
# measurements is poorly defined. 
@njit(cache=True, nogil=True)
def log_prior_hierarchical_5d_ncp(theta):
    # Load in the values that define our parameter space. 
    log_alpha_true, eta_x, eta_y, gamma_int_x, gamma_int_y = theta

    # Set the physical limits of the domain. 
    if not (-2.0 < log_alpha_true < 2.0): return -np.inf

    # Define the y-intercept.
    if not (-100 < gamma_int_x < 100): return -np.inf
    if not (-100 < gamma_int_y < 100): return -np.inf

    # Draw the eta values from the a standard normal N(0, 1)
    # then in the log_likelihood_hierarchical_5d_ncp defines the stretching 
    # factor to build the asymmetric prior. 
    log_P_eta_x = -0.5 * eta_x**2       # Note, mu = 0 and sigma = 1. Dropping the 1/sqrt(2pi) term.
    log_P_eta_y = -0.5 * eta_y**2

    # Boundaries of the intrinsic velocity fluctuation offset.
    sigma_gamma = 15
    log_P_gamma_x = -0.5 * (gamma_int_x / sigma_gamma)**2
    log_P_gamma_y = -0.5 * (gamma_int_y / sigma_gamma)**2

    # The physical informative prior
    log_P_physical = 0.0
    penalty_steepness = 0.05  
    if log_alpha_true > 1.0:
        log_P_physical = -0.5 * ((log_alpha_true - 1.0) / penalty_steepness)**2
    elif log_alpha_true < -1.0:
        log_P_physical = -0.5 * ((log_alpha_true - (-1.0)) / penalty_steepness)**2

    log_prior = log_P_eta_x + log_P_eta_y + log_P_gamma_x + log_P_gamma_y + log_P_physical

    return log_prior

@njit(fastmath=True, cache=True, nogil=True)
def log_likelihood_hierarchical_5d_ncp(theta, vzbx, vx, vzby, vy, bz, var_x_array, var_y_array,
                                       ani_list, dp_slice_x, dm_slice_x, bias_slice_x, 
                                       dp_slice_y, dm_slice_y, bias_slice_y, 
                                       use_t, use_gmm, weights_x, weights_y):
    # Load in the values that define our parameter space. 
    log_alpha_true, eta_x, eta_y, gamma_int_x, gamma_int_y = theta
    log_ani_list = np.log10(ani_list)

    # 1. Global Boundary Check
    idx = np.searchsorted(log_ani_list, log_alpha_true)
    if idx == 0 or idx == len(log_ani_list):
        return -np.inf      
    
    # 2. X-Plane NaN Bracket Check (Added Bias)
    if np.isnan(dp_slice_x[idx - 1]) or np.isnan(dp_slice_x[idx]) or \
       np.isnan(dm_slice_x[idx - 1]) or np.isnan(dm_slice_x[idx]) or \
       np.isnan(bias_slice_x[idx - 1]) or np.isnan(bias_slice_x[idx]):
        return -np.inf
    
    # 3. Y-Plane NaN Bracket Check (Added Bias)
    if np.isnan(dp_slice_y[idx - 1]) or np.isnan(dp_slice_y[idx]) or \
       np.isnan(dm_slice_y[idx - 1]) or np.isnan(dm_slice_y[idx]) or \
       np.isnan(bias_slice_y[idx - 1]) or np.isnan(bias_slice_y[idx]):
        return -np.inf

    # 4. X-Plane Evaluation (Masks removed for maximum JIT speed)
    log_dp_x = np.interp(log_alpha_true, log_ani_list, dp_slice_x)
    log_dm_x = np.interp(log_alpha_true, log_ani_list, dm_slice_x)
    log_bias_x = np.interp(log_alpha_true, log_ani_list, bias_slice_x)

    expected_x = log_alpha_true + log_bias_x
    sig_x = log_dp_x if eta_x > 0 else log_dm_x     
    sig_x = max(sig_x, 0.001)       

    log_alpha_eff_x = expected_x + (eta_x * sig_x)

    # 5. Y-Plane Evaluation (Masks removed for maximum JIT speed)
    log_dp_y = np.interp(log_alpha_true, log_ani_list, dp_slice_y)
    log_dm_y = np.interp(log_alpha_true, log_ani_list, dm_slice_y)
    log_bias_y = np.interp(log_alpha_true, log_ani_list, bias_slice_y)

    expected_y = log_alpha_true + log_bias_y
    sig_y = log_dp_y if eta_y > 0 else log_dm_y
    sig_y = max(sig_y, 0.001)

    log_alpha_eff_y = expected_y + (eta_y * sig_y)

    # 6. Final Computations
    log_alpha_eff_x = max(min(log_alpha_eff_x, 5.0), -5.0)
    log_alpha_eff_y = max(min(log_alpha_eff_y, 5.0), -5.0)

    alpha_eff_x = 10**(log_alpha_eff_x)
    alpha_eff_y = 10**(log_alpha_eff_y)

    denom_x = max(((bz**2) + alpha_eff_x * (1.0 - bz**2)), 1e-30)
    denom_y = max(((bz**2) + alpha_eff_y * (1.0 - bz**2)), 1e-30)

    M_x = ((1.0 - alpha_eff_x) * bz) / denom_x  
    M_y = ((1.0 - alpha_eff_y) * bz) / denom_y 
    
    mu_x = M_x * vzbx + gamma_int_x
    mu_y = M_y * vzby + gamma_int_y
    
    chi_sq_x = ((vx - mu_x)**2) / var_x_array
    chi_sq_y = ((vy - mu_y)**2) / var_y_array

    if use_gmm:
        kappa_sq = 100.0
        kappa = 10.0

        eta_mix_x = 1.0 - weights_x
        p1_x = (1.0 - eta_mix_x) * np.exp(-0.5 * chi_sq_x) / np.sqrt(var_x_array)
        p2_x = eta_mix_x * np.exp(-0.5 * chi_sq_x / kappa_sq) / (kappa * np.sqrt(var_x_array))
        L_x = -np.sum(np.log(p1_x + p2_x + 1e-30))   # Add a small buffer if values are zeros

        eta_mix_y = 1.0 - weights_y
        p1_y = (1.0 - eta_mix_y) * np.exp(-0.5 * chi_sq_y) / np.sqrt(var_y_array)
        p2_y = eta_mix_y * np.exp(-0.5 * chi_sq_y / kappa_sq) / (kappa * np.sqrt(var_y_array))
        L_y = -np.sum(np.log(p1_y + p2_y + 1e-30))
    elif use_t:
        # Student-t likelihood with nu = 4
        L_x = np.sum(2.5 * np.log(1.0 + chi_sq_x / 4.0) + 0.5 * np.log(var_x_array))
        L_y = np.sum(2.5 * np.log(1.0 + chi_sq_y / 4.0) + 0.5 * np.log(var_y_array))
    else:
        L_x = 0.5 * np.sum(chi_sq_x + np.log(var_x_array))  
        L_y = 0.5 * np.sum(chi_sq_y + np.log(var_y_array))
        
    return -(L_x + L_y)

@njit(cache=True, nogil=True)
def log_probability_hierarchical_5d_ncp(theta, vzbx, vx, vzby, vy, bz, ani_list, 
                                        dp_slice_x, dm_slice_x, bias_slice_x,
                                        dp_slice_y, dm_slice_y, bias_slice_y,
                                        var_x_array, var_y_array, use_t, use_gmm,
                                        weights_x, weights_y):
    # Evaluate the log prior
    lp = log_prior_hierarchical_5d_ncp(theta)

    # Quich check to make sure the log prior is finite
    if not np.isfinite(lp):
        return -np.inf
    
    # Evaluate the log likelihood
    ll = log_likelihood_hierarchical_5d_ncp(theta, vzbx, vx, vzby, vy, bz, 
                                            var_x_array, var_y_array, ani_list, 
                                            dp_slice_x, dm_slice_x, bias_slice_x,
                                            dp_slice_y, dm_slice_y, bias_slice_y,
                                            use_t, use_gmm, weights_x, weights_y)
    
    # Quick check to make sure the log likelihood is finite
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

def reconstruct_effective_chain(log_true_chain, eta_chain, log_true_lookup, log_ani_list, dp_slice, dm_slice, bias_slice):
    """
    Reconstructs the physical effective log-anisotropy chain from the NCP eta chain 
    using the deterministic split-normal transformation.
    """
    valid = np.isfinite(dp_slice) & np.isfinite(dm_slice) & np.isfinite(bias_slice)
    
    # If the uncertainty map slice is completely empty/invalid, fallback to the true state
    if np.sum(valid) < 2:
        return log_true_chain
        
    # Interpolate the uncertainty map components
    log_dp = np.interp(log_true_lookup, log_ani_list[valid], dp_slice[valid])
    log_dm = np.interp(log_true_lookup, log_ani_list[valid], dm_slice[valid])
    log_bias = np.interp(log_true_lookup, log_ani_list[valid], bias_slice[valid])
    
    # Apply the split-normal logic based on the sign of the eta chain
    sig_chain = np.where(eta_chain > 0, log_dp, log_dm)
    sig_chain = np.maximum(sig_chain, 0.01)
    
    # Transform back to the effective physical state
    log_eff_chain = (log_true_chain + log_bias) + (eta_chain * sig_chain)
    
    return log_eff_chain

def _get_uncertainty_slices(angle, uncert_map):
    """
    Helper function that interpolates and validates the 1D Slices from the 2D Uncertainty Map.
    """

    angle_list, ani_list, dp_map, dm_map, bias_map = uncert_map
    idx = np.nanargmin(np.abs(np.abs(angle) - angle_list))

    dp_slice, dm_slice, bias_slice = dp_map[:, idx], dm_map[:, idx], bias_map[:, idx]
    valid = np.isfinite(dp_slice) & np.isfinite(dm_slice) & np.isfinite(bias_slice)

    log_ani = np.log10(ani_list)
    safe_min = log_ani[valid][0] if np.any(valid) else -np.inf
    safe_max = log_ani[valid][-1] if np.any(valid) else np.inf

    return dp_slice, dm_slice, bias_slice, safe_min, safe_max

def _get_percentile_errors(data):
    """
    Compute the median, linear, and logarithmic uncertainties from a 1D array of data.
    Returns:
    -------
    p50 : float
        The median of the data.
    plus_lin : float
        The linear uncertainty above the median (p84 - p50).
    minus_lin : float
        The linear uncertainty below the median (p50 - p16).
    plus_log : float
        The logarithmic uncertainty above the median (log10(p84) - log10(p50)).
    minus_log : float
        The logarithmic uncertainty below the median (log10(p50) - log10(p16)).
    """
    p16, p50, p84 = np.percentile(data, [16, 50, 84])
    plus_lin = p84 - p50
    minus_lin = p50 - p16
    if p16 > 0 and p50 > 0 and p84 > 0:
        plus_log = np.log10(p84) - np.log10(p50)
        minus_log = np.log10(p50) - np.log10(p16)
    else:
        plus_log, minus_log = np.nan, np.nan
    return p50, plus_lin, minus_lin, plus_log, minus_log

DEFAULT_RES_BUNDLE = {
            'alpha': np.nan, 'sig_plus_lin': np.nan, 'sig_minus_lin': np.nan,
            'sig_plus_log': np.nan, 'sig_minus_log': np.nan,
            'alpha_eff_x': np.nan, 'sig_plus_x': np.nan, 'sig_minus_x': np.nan,
            'sig_plus_log_x': np.nan, 'sig_minus_log_x': np.nan,
            'alpha_eff_y': np.nan, 'sig_plus_y': np.nan, 'sig_minus_y': np.nan,
            'sig_plus_log_y': np.nan, 'sig_minus_log_y': np.nan,
            'gamma_x': np.nan, 'sig_plus_gam_x': np.nan, 'sig_minus_gam_x': np.nan,
            'sig_plus_log_gam_x': np.nan, 'sig_minus_log_gam_x': np.nan,
            'gamma_y': np.nan, 'sig_plus_gam_y': np.nan, 'sig_minus_gam_y': np.nan,
            'sig_plus_log_gam_y': np.nan, 'sig_minus_log_gam_y': np.nan,
            'samples': np.full((1, 5), np.nan)
        }

def run_emcee_hierarchical_5d_ncp(vzbx, vx, vzby, vy, bz, uncert_map, 
                                  xangle, yangle, var_x_array, var_y_array, 
                                  likelihood_type = "Gauss", mix_weights = None,
                                  nwalkers = 16, nsteps = 400, burn_in = 50):
    use_t = True if likelihood_type.lower() == "t" else False
    use_gmm = True if likelihood_type.lower() == "gmm" else False

    if use_gmm and mix_weights is None:
        raise ValueError("Weights must be provided when using the GMM likelihood method.")

    mask_x = np.isfinite(vzbx) & np.isfinite(vx) & np.isfinite(var_x_array)
    vzbx, vx, var_x = vzbx[mask_x], vx[mask_x], var_x_array[mask_x]
    
    mask_y = np.isfinite(vzby) & np.isfinite(vy) & np.isfinite(var_y_array)
    vzby, vy, var_y = vzby[mask_y], vy[mask_y], var_y_array[mask_y]

    # I do not know what these are...
    w_x = mix_weights[mask_x] if mix_weights is not None else np.zeros_like(var_x)
    w_y = mix_weights[mask_y] if mix_weights is not None else np.zeros_like(var_y)

    if len(vzbx) < 2 or len(vzby) < 2:
        raise ValueError("Not enough finite data points to run 5D MCMC.")

    # Calculate Initial Alpha Guesses
    emp_alpha_x, slope_x, intercept_x, weight_x = _calculate_empirical_guess(vzbx, vx, var_x, bz)
    emp_alpha_y, slope_y, intercept_y, weight_y = _calculate_empirical_guess(vzby, vy, var_y, bz)

    alphas = np.array([emp_alpha_x, emp_alpha_y])
    aweights = np.array([weight_x, weight_y])
    valid = np.isfinite(alphas) & (alphas > 0) & (aweights > 0)
    
    if np.any(valid):
        emp_alpha = np.average(alphas[valid], weights=aweights[valid])
    else:
        emp_alpha = 1.0

    emp_alpha = np.clip(emp_alpha, 0.15, 9.85)
    log_emp_alpha = np.log10(emp_alpha)
    
    # Set the Initial Gamma (Intercept) Guesses from the empricial guess
    gamma_guess_x = 0.0 # intercept_x if np.isfinite(intercept_x) else 0.0
    gamma_guess_y = 0.0 # intercept_y if np.isfinite(intercept_y) else 0.0

    # Handle Uncertainty Map Lookup
    dp_x, dm_x, bias_x, min_x, max_x = _get_uncertainty_slices(xangle, uncert_map)
    dp_y, dm_y, bias_y, min_y, max_y = _get_uncertainty_slices(yangle, uncert_map)

    safe_min = max(min_x, min_y) + 0.01
    safe_max = min(max_x, max_y) - 0.01

    if safe_min >= safe_max:
        raise ValueError("Safe min and max for log_alpha_true are invalid. Check uncertainty map slices.")
    
    # 4. Initialize the 5D Walkers
    ndim = 5        

    # Initialize the alpha true walkers as a random variable number added to the empirical guess.
    log_pos_alpha_true = log_emp_alpha + 1e-2 * np.random.randn(nwalkers)
    log_pos_alpha_true = np.clip(log_pos_alpha_true, safe_min, safe_max)

    # Initialize the eta parameters. For this we will start at zero!
    pos_eta_x = 0.0 + 1e-2 * np.random.randn(nwalkers)
    pos_eta_y = 0.0 + 1e-2 * np.random.randn(nwalkers)

    # Define the initial intercept values
    pos_gamma_x = gamma_guess_x + 0.1 * np.random.randn(nwalkers)
    pos_gamma_y = gamma_guess_y + 0.1 * np.random.randn(nwalkers)

    # Our initial parameter space!
    pos = np.vstack((log_pos_alpha_true, pos_eta_x, pos_eta_y, pos_gamma_x, pos_gamma_y)).T

    res_bundle = DEFAULT_RES_BUNDLE.copy()
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_hierarchical_5d_ncp,
                                        args = (vzbx, vx, vzby, vy, bz, uncert_map[1], 
                                        dp_x, dm_x, bias_x, dp_y, dm_y, bias_y, 
                                        var_x, var_y, use_t, use_gmm, w_x, w_y) 
                )

        sampler.run_mcmc(pos, nsteps, progress=False)

        log_samples = sampler.get_chain(discard=burn_in, flat=True)

        ani_list = uncert_map[1]
        log_ani_list = np.log10(ani_list)

        log_true = log_samples[:, 0]
        log_lookup = np.clip(log_true, log_ani_list[0], log_ani_list[-1])
        
        eta_x_chain    = log_samples[:, 1]
        eta_y_chain    = log_samples[:, 2]
        gamma_x_chain  = log_samples[:, 3]
        gamma_y_chain  = log_samples[:, 4]

        log_eff_x_chain = reconstruct_effective_chain(log_true, eta_x_chain, log_lookup, 
                                                      log_ani_list, dp_x, dm_x, bias_x)
        log_eff_y_chain = reconstruct_effective_chain(log_true, eta_y_chain, log_lookup, 
                                                      log_ani_list, dp_y, dm_y, bias_y)
        
        samples_true  = 10**np.clip(log_true, -5.0, 5.0)    # Add in clipping to avoid NaN values in power
        samples_eff_x = 10**np.clip(log_eff_x_chain, -5.0, 5.0) # Add in clipping to avoid NaN values in power
        samples_eff_y = 10**np.clip(log_eff_y_chain, -5.0, 5.0) # Add in clipping to avoid NaN values in power

        # Compute the median and uncertainties for each parameter
        a_med, a_plus_lin, a_minus_lin, a_plus_log, a_minus_log = _get_percentile_errors(samples_true)
        ax_med, ax_plus, ax_minus, ax_plus_log, ax_minus_log = _get_percentile_errors(samples_eff_x)
        ay_med, ay_plus, ay_minus, ay_plus_log, ay_minus_log = _get_percentile_errors(samples_eff_y)
        gx_med, gx_plus, gx_minus, gx_plus_log, gx_minus_log = _get_percentile_errors(gamma_x_chain)
        gy_med, gy_plus, gy_minus, gy_plus_log, gy_minus_log = _get_percentile_errors(gamma_y_chain)

        samples = np.vstack((samples_true, samples_eff_x, samples_eff_y, gamma_x_chain, gamma_y_chain)).T

        res_bundle.update({
            'alpha': a_med, 'sig_plus_lin': a_plus_lin, 'sig_minus_lin': a_minus_lin,
            'sig_plus_log': a_plus_log, 'sig_minus_log': a_minus_log,
            'alpha_eff_x': ax_med, 'sig_plus_x': ax_plus, 'sig_minus_x': ax_minus,
            'sig_plus_log_x': ax_plus_log, 'sig_minus_log_x': ax_minus_log,
            'alpha_eff_y': ay_med, 'sig_plus_y': ay_plus, 'sig_minus_y': ay_minus,
            'sig_plus_log_y': ay_plus_log, 'sig_minus_log_y': ay_minus_log,
            'gamma_x': gx_med, 'sig_plus_gam_x': gx_plus, 'sig_minus_gam_x': gx_minus,
            'sig_plus_log_gam_x': gx_plus_log, 'sig_minus_log_gam_x': gx_minus_log,
            'gamma_y': gy_med, 'sig_plus_gam_y': gy_plus, 'sig_minus_gam_y': gy_minus,
            'sig_plus_log_gam_y': gy_plus_log, 'sig_minus_log_gam_y': gy_minus_log,
            'samples': samples
        })
    except Exception as e:
        print(f"Error during MCMC sampling: {e}")
        res_bundle = DEFAULT_RES_BUNDLE.copy()
        pass

    return res_bundle

def _run_bayesian_pipeline_5d_ncp(block, valid_idx, vzbx_fit, vx_fit, 
                              vzby_fit, vy_fit, vz_var, block_nvec_avg,
                              uncertainty_map, xangle, yangle, static_length, 
                              likelihood_type='Gauss'):

    lamda = 1.009       
    C = lamda/2.0

    # Grab in the plate currents that will be used for defining the plate differences
    ia_fit = block['ia_curr'][valid_idx]
    ib_fit = block['ib_curr'][valid_idx]
    ic_fit = block['ic_curr'][valid_idx]
    id_fit = block['id_curr'][valid_idx]

    i_tot = np.maximum(ia_fit + ib_fit + ic_fit + id_fit, 1e-6)     
    i_diff_x = (ia_fit + id_fit) - (ib_fit + ic_fit)
    i_diff_y = (ia_fit + ib_fit) - (ic_fit + id_fit)

    # Convert the standard deviations to variances
    var_A = block['ia_std'][valid_idx]**2
    var_B = block['ib_std'][valid_idx]**2
    var_C = block['ic_std'][valid_idx]**2
    var_D = block['id_std'][valid_idx]**2

    # X-Z Plane: Positive = A, D | Negative = B, C
    Sigma_pos_x_sq = var_A + var_D
    Sigma_neg_x_sq = var_B + var_C

    # Y-Z Plane: Positive = A, B | Negative = C, D
    Sigma_pos_y_sq = var_A + var_B
    Sigma_neg_y_sq = var_C + var_D

    # Get the raw vx and vy values
    vx_raw = block['vx'][valid_idx]
    vy_raw = block['vy'][valid_idx]
    vz_raw = block['vz'][valid_idx] 
    vz_raw_abs = np.abs(vz_raw)

    # This is the tangent term
    tan_theta_x = vx_raw / vz_raw_abs    
    tan_theta_y = vy_raw / vz_raw_abs

    # Get the angle in radians
    theta_x_rad = np.arctan(tan_theta_x)
    theta_y_rad = np.arctan(tan_theta_y)

    # Define the angular variance prefix
    angle_var_prefix = (lamda / (2.0 * i_tot**2))**2

    # The variance in theta x
    var_theta_x = angle_var_prefix * (
        ((i_tot - i_diff_x)**2 * Sigma_pos_x_sq) + 
        ((i_tot + i_diff_x)**2 * Sigma_neg_x_sq)
    )

    # The variance in theta y
    var_theta_y = angle_var_prefix * (
        ((i_tot - i_diff_y)**2 * Sigma_pos_y_sq) + 
        ((i_tot + i_diff_y)**2 * Sigma_neg_y_sq)
    )

    # The geometric modifier
    geom_mod_x = (vz_raw_abs / np.cos(theta_x_rad)**2)**2
    geom_mod_y = (vz_raw_abs / np.cos(theta_y_rad)**2)**2

    term2_x = geom_mod_x * var_theta_x
    term2_y = geom_mod_y * var_theta_y

    # norm_weights = i_tot / np.sum(i_tot)
    term1_x = (tan_theta_x**2) * vz_var
    term1_y = (tan_theta_y**2) * vz_var

    # The variance of the bulk velocity is the sum of the 
    # squared normalized weights * the measurement variances
    # term3_x = np.sum((norm_weights**2) * term2_x)
    # term3_y = np.sum((norm_weights**2) * term2_y)

    var_x_array_inst = np.maximum(term1_x + term2_x, 1e-4) # + term3_x, 1e-4)
    var_y_array_inst = np.maximum(term1_y + term2_y, 1e-4) # + term3_y, 1e-4)

    safe_weights = np.maximum(block['weight_fits'] if 'weight_fits' in block else 1.0, 1e-4)

    var_x_array = var_x_array_inst / safe_weights
    var_y_array = var_y_array_inst / safe_weights

    weights_rdf = block['weights_rdf'][valid_idx]

    # Call the NCP EMCEE sampler
    res_bundle = run_emcee_hierarchical_5d_ncp(
        vzbx_fit, vx_fit, vzby_fit, vy_fit, block_nvec_avg[2], 
        uncertainty_map, xangle, yangle, var_x_array, var_y_array, 
        likelihood_type, mix_weights=weights_rdf
    )

    # Unpack from res_bundle using the updated keys
    avg_alpha         = res_bundle['alpha']
    avg_sig_plus_lin  = res_bundle['sig_plus_lin']
    avg_sig_minus_lin = res_bundle['sig_minus_lin']
    avg_sig_plus_log  = res_bundle['sig_plus_log']
    avg_sig_minus_log = res_bundle['sig_minus_log']

    alpha_eff_x      = res_bundle['alpha_eff_x']
    sig_plus_x_lin   = res_bundle['sig_plus_x']
    sig_minus_x_lin  = res_bundle['sig_minus_x']
    sig_plus_x_log   = res_bundle['sig_plus_log_x']
    sig_minus_x_log  = res_bundle['sig_minus_log_x']

    alpha_eff_y      = res_bundle['alpha_eff_y']
    sig_plus_y_lin   = res_bundle['sig_plus_y']
    sig_minus_y_lin  = res_bundle['sig_minus_y']
    sig_plus_y_log   = res_bundle['sig_plus_log_y']
    sig_minus_y_log  = res_bundle['sig_minus_log_y']

    gamma_x          = res_bundle['gamma_x']
    sig_plus_gam_x   = res_bundle['sig_plus_gam_x']
    sig_minus_gam_x  = res_bundle['sig_minus_gam_x']

    gamma_y          = res_bundle['gamma_y']
    sig_plus_gam_y   = res_bundle['sig_plus_gam_y']
    sig_minus_gam_y  = res_bundle['sig_minus_gam_y']

    samples          = res_bundle['samples']

    bz = block_nvec_avg[2]
    dof_single = max(len(vzbx_fit) - 2.0, 1.0)
    dof_joint = max((2.0 * len(vzbx_fit)) - 3.0, 1.0)

    if np.isfinite(avg_alpha):
        denom_joint = max(((bz**2) + avg_alpha * (1.0 - bz**2)), 1e-12)
        m_joint = ((1.0 - avg_alpha) * bz) / denom_joint

        m_eff_x = ((1.0 - alpha_eff_x) * bz) / max(((bz**2) + alpha_eff_x * (1.0 - bz**2)), 1e-12)
        m_eff_y = ((1.0 - alpha_eff_y) * bz) / max(((bz**2) + alpha_eff_y * (1.0 - bz**2)), 1e-12)

        sq_res_x = ((vx_fit - (m_joint * vzbx_fit + gamma_x))**2) / var_x_array
        sq_res_y = ((vy_fit - (m_joint * vzby_fit + gamma_y))**2) / var_y_array

        sq_res_eff_x = ((vx_fit - (m_eff_x * vzbx_fit + gamma_x))**2) / var_x_array
        sq_res_eff_y = ((vy_fit - (m_eff_y * vzby_fit + gamma_y))**2) / var_y_array

        # Standard reduced chi-squared
        chi_sq_x = np.sum(sq_res_x) / dof_single
        chi_sq_y = np.sum(sq_res_y) / dof_single
        chi_sq_joint = (np.sum(sq_res_x) + np.sum(sq_res_y)) / dof_joint

        # Robust Student-t chi-squared: weights w_i = (nu + 1) / (nu + r_i^2) with nu = 4
        w_rob_x = 5.0 / (4.0 + sq_res_x)
        w_rob_y = 5.0 / (4.0 + sq_res_y)
        chi_sq_x_robust = np.sum(w_rob_x * sq_res_x) / dof_single
        chi_sq_y_robust = np.sum(w_rob_y * sq_res_y) / dof_single
        chi_sq_joint_robust = (np.sum(w_rob_x * sq_res_x) + np.sum(w_rob_y * sq_res_y)) / dof_joint

        chi_sq_eff_x = np.sum(sq_res_eff_x) / dof_single
        chi_sq_eff_y = np.sum(sq_res_eff_y) / dof_single
    else:
        chi_sq_x = chi_sq_y = chi_sq_joint = np.nan
        chi_sq_x_robust = chi_sq_y_robust = chi_sq_joint_robust = np.nan
        chi_sq_eff_x = chi_sq_eff_y = np.nan

    # Get the effective x and y fits.
    # m_x = ((1.0 - avg_alpha) * bz) / (1.0 - (1.0 - avg_alpha) * (1.0 - bz**2))
    # m_y = ((1.0 - avg_alpha) * bz) / (1.0 - (1.0 - avg_alpha) * (1.0 - bz**2))

    # m_eff_x = ((1.0 - alpha_eff_x) * bz) / (1.0 - (1.0 - alpha_eff_x) * (1.0 - bz**2))
    # m_eff_y = ((1.0 - alpha_eff_y) * bz) / (1.0 - (1.0 - alpha_eff_y) * (1.0 - bz**2))
    
    # sq_res_x_valid = ((vx_fit - (m_x * vzbx_fit + gamma_x))**2)/ var_x_array
    # sq_res_y_valid = ((vy_fit - (m_y * vzby_fit + gamma_y))**2)/ var_y_array

    # sq_res_eff_x_valid = ((vx_fit - (m_eff_x * vzbx_fit + gamma_x))**2)/ var_x_array
    # sq_res_eff_y_valid = ((vy_fit - (m_eff_y * vzby_fit + gamma_y))**2)/ var_y_array

    # dof_single = max(len(vzbx_fit) - 2.0, 1.0)
    # dof_joint = max((2.0 * len(vzbx_fit)) - 3.0, 1.0)

    # chi_sq_x = np.sum(sq_res_x_valid) / dof_single
    # chi_sq_y = np.sum(sq_res_y_valid) / dof_single
    # chi_sq_joint = (np.sum(sq_res_x_valid) + np.sum(sq_res_y_valid)) / dof_joint

    # # Cauchy Pseudo-Chi-Squared Calculations
    # chi_sq_x_cauchy = np.sum(2.0 * np.log(1.0 + 0.5 * sq_res_x_valid)) / dof_single
    # chi_sq_y_cauchy = np.sum(2.0 * np.log(1.0 + 0.5 * sq_res_y_valid)) / dof_single
    # chi_sq_joint_cauchy = (dof_single * chi_sq_x_cauchy + dof_single * chi_sq_y_cauchy) / dof_joint

    # chi_sq_eff_x = np.sum(sq_res_eff_x_valid) / dof_single
    # chi_sq_eff_y = np.sum(sq_res_eff_y_valid) / dof_single

    # # Create an emply 2D array shaped exactly like the input block (time, dim)
    # block_shape = valid_idx.shape
    # block_res_x = np.full(block_shape, np.nan)
    # block_res_y = np.full(block_shape, np.nan)
    
    # # Map the 1D valid residuals back into their exact physical velocity bins
    # block_res_x[valid_idx] = sq_res_x_valid
    # block_res_y[valid_idx] = sq_res_y_valid 

    update_dict = {
        # Joint consensus temperature anisotropy
        "T_ani_avg": avg_alpha,
        "T_ani_avg_delta_plus": avg_sig_plus_lin,
        "T_ani_avg_delta_minus": avg_sig_minus_lin,
        "T_ani_avg_delta_plus_log": avg_sig_plus_log,
        "T_ani_avg_delta_minus_log": avg_sig_minus_log,

        # Single-plane effective anisotropies
        "T_ani_chi": alpha_eff_x,
        "T_ani_chi_delta_plus": sig_plus_x_lin,
        "T_ani_chi_delta_minus": sig_minus_x_lin,
        "T_ani_chi_delta_plus_log": sig_plus_x_log,
        "T_ani_chi_delta_minus_log": sig_minus_x_log,

        "T_ani_psi": alpha_eff_y,
        "T_ani_psi_delta_plus": sig_plus_y_lin,
        "T_ani_psi_delta_minus": sig_minus_y_lin,
        "T_ani_psi_delta_plus_log": sig_plus_y_log,
        "T_ani_psi_delta_minus_log": sig_minus_y_log,

        # Velocity intercepts
        "gamma_chi": gamma_x,
        "gamma_chi_delta_plus": sig_plus_gam_x,
        "gamma_chi_delta_minus": sig_minus_gam_x,
        "gamma_psi": gamma_y,
        "gamma_psi_delta_plus": sig_plus_gam_y,
        "gamma_psi_delta_minus": sig_minus_gam_y,

        # Standard Chi-squared statistics
        "chi_sq_joint": chi_sq_joint,
        "chi_sq_chi": chi_sq_x,
        "chi_sq_psi": chi_sq_y,
        "chi_sq_eff_x": chi_sq_eff_x,
        "chi_sq_eff_y": chi_sq_eff_y,

        # Robust Student-t Chi-squared statistics
        "chi_sq_joint_robust": chi_sq_joint_robust,
        "chi_sq_chi_robust": chi_sq_x_robust,
        "chi_sq_psi_robust": chi_sq_y_robust,
    }

    # We also return samples and var arrays so the main loop can handle plotting!
    return update_dict, samples, var_x_array, var_y_array

@njit(cache=True, nogil=True)
def log_prior_hierarchical_3d_ncp(theta):
    log_alpha_true, eta, gamma_int = theta

    if not (-2.0 < log_alpha_true < 2.0): return -np.inf
    if not (-100 < gamma_int < 100): return -np.inf

    log_P_eta = -0.5 * eta**2
    log_P_gamma = -0.5 * (gamma_int / 15.0)**2

    log_P_physical = 0.0
    penalty_steepness = 0.05  
    if log_alpha_true > 1.0:
        log_P_physical = -0.5 * ((log_alpha_true - 1.0) / penalty_steepness)**2
    elif log_alpha_true < -1.0:
        log_P_physical = -0.5 * ((log_alpha_true - (-1.0)) / penalty_steepness)**2

    return log_P_eta + log_P_gamma + log_P_physical

@njit(fastmath=True, cache=True, nogil=True)
def log_likelihood_hierarchical_3d_ncp(theta, vzb, v_trans, bz, var_array,
                                       ani_list, dp_slice, dm_slice, bias_slice,
                                       use_t, use_gmm, weights):
    log_alpha_true, eta, gamma_int = theta
    log_ani_list = np.log10(ani_list)

    idx = np.searchsorted(log_ani_list, log_alpha_true)
    if idx == 0 or idx == len(log_ani_list):
        return -np.inf      
    
    if np.isnan(dp_slice[idx - 1]) or np.isnan(dp_slice[idx]) or \
       np.isnan(dm_slice[idx - 1]) or np.isnan(dm_slice[idx]) or \
       np.isnan(bias_slice[idx - 1]) or np.isnan(bias_slice[idx]):
        return -np.inf

    log_dp = np.interp(log_alpha_true, log_ani_list, dp_slice)
    log_dm = np.interp(log_alpha_true, log_ani_list, dm_slice)
    log_bias = np.interp(log_alpha_true, log_ani_list, bias_slice)

    expected = log_alpha_true + log_bias
    sig = log_dp if eta > 0 else log_dm     
    sig = max(sig, 0.001)       

    log_alpha_eff = expected + (eta * sig)
    log_alpha_eff = max(min(log_alpha_eff, 5.0), -5.0)
    alpha_eff = 10**(log_alpha_eff)

    denom = max(((bz**2) + alpha_eff * (1.0 - bz**2)), 1e-30)
    M = ((1.0 - alpha_eff) * bz) / denom  
    mu = M * vzb + gamma_int
    
    chi_sq = ((v_trans - mu)**2) / var_array
    
    # NOTE: You mentioned adding a Cauchy likelihood. If you pass a boolean flag 
    # to this function, you can toggle between Gaussian (below) or Cauchy here!
    if use_gmm:
        kappa_sq = 100.0
        kappa = 10.0

        eta_mix = 1.0 - weights
        p1 = (1.0 - eta_mix) * np.exp(-0.5 * chi_sq) / np.sqrt(var_array)
        p2 = eta_mix * np.exp(-0.5 * chi_sq / kappa_sq) / (kappa * np.sqrt(var_array))
        L = -np.sum(np.log(p1 + p2 + 1e-30)) # Add a small buffer.
    elif use_t:
        # Student-t likelihood with nu = 4
        L = np.sum(2.5 * np.log(1.0 + chi_sq / 4.0) + 0.5 * np.log(var_array))
    else:
        L = 0.5 * np.sum(chi_sq + np.log(var_array))

    return -1.0 * L

@njit(cache=True, nogil=True)
def log_probability_hierarchical_3d_ncp(theta, vzb, v_trans, bz, ani_list, 
                                        dp_slice, dm_slice, bias_slice, var_array, 
                                        use_t, use_gmm, weights):
    lp = log_prior_hierarchical_3d_ncp(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood_hierarchical_3d_ncp(theta, vzb, v_trans, bz, var_array, 
                                            ani_list, dp_slice, dm_slice, bias_slice, 
                                            use_t, use_gmm, weights)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

def _run_bayesian_pipeline_3d_ncp(block, valid_idx, vzb_fit, v_trans_fit, vz_var, 
                                  block_nvec_avg, uncertainty_map, flow_angle, 
                                  static_length, tag='x', likelihood_type='t',
                                  nwalkers=16, nsteps=400, burn_in=50):
    
    use_t = True if likelihood_type.lower() == 't' else False
    use_gmm = True if likelihood_type.lower() == 'gmm' else False

    lamda = 1.009
    # 1. Calculate the single-plane variance (simplified from 5D)
    # Grab in the plate currents that will be used for defining the plate differences
    ia_fit = block['ia_curr'][valid_idx]
    ib_fit = block['ib_curr'][valid_idx]
    ic_fit = block['ic_curr'][valid_idx]
    id_fit = block['id_curr'][valid_idx]

    i_tot = np.maximum(ia_fit + ib_fit + ic_fit + id_fit, 1e-6)     
    i_diff_x = (ia_fit + id_fit) - (ib_fit + ic_fit)
    i_diff_y = (ia_fit + ib_fit) - (ic_fit + id_fit)

    # Convert the standard deviations to variances
    var_A = block['ia_std'][valid_idx]**2
    var_B = block['ib_std'][valid_idx]**2
    var_C = block['ic_std'][valid_idx]**2
    var_D = block['id_std'][valid_idx]**2

    # X-Z Plane: Positive = A, D | Negative = B, C
    Sigma_pos_x_sq = var_A + var_D
    Sigma_neg_x_sq = var_B + var_C

    # Y-Z Plane: Positive = A, B | Negative = C, D
    Sigma_pos_y_sq = var_A + var_B
    Sigma_neg_y_sq = var_C + var_D
    
    # Get the raw vx and vy values
    vx_raw = block['vx'][valid_idx]
    vy_raw = block['vy'][valid_idx]
    vz_raw = block['vz'][valid_idx] 
    vz_raw_abs = np.abs(vz_raw)

    # This is the tangent term
    # tan_theta_x = vx_raw / vz_raw_abs    
    # tan_theta_y = vy_raw / vz_raw_abs

    # Get the angle in radians
    # theta_x_rad = np.arctan(tan_theta_x)
    # theta_y_rad = np.arctan(tan_theta_y)

    # Define the angular variance prefix
    angle_var_prefix = (lamda / (2.0 * i_tot**2))**2

    if tag == 'x':
        tan_theta = vx_raw / vz_raw_abs    
        theta_rad = np.arctan(tan_theta)
        var_theta = angle_var_prefix * (
            ((i_tot - i_diff_x)**2 * Sigma_pos_x_sq) + 
            ((i_tot + i_diff_x)**2 * Sigma_neg_x_sq)
        )
    else:
        tan_theta = vy_raw / vz_raw_abs
        theta_rad = np.arctan(tan_theta)
        var_theta = angle_var_prefix * (
            ((i_tot - i_diff_y)**2 * Sigma_pos_y_sq) + 
            ((i_tot + i_diff_y)**2 * Sigma_neg_y_sq)
        )

    # The geometric modifier
    geom_mod = (vz_raw_abs / np.cos(theta_rad)**2)**2
    term2 = geom_mod * var_theta
    term1 = (tan_theta**2) * vz_var

    safe_weights = np.maximum(block['weight_fits'] if 'weight_fits' in block else 1.0, 1e-4)
    var_array = np.maximum(term1 + term2, 1e-4) / safe_weights
    weights_rdf = block['weights_rdf'][valid_idx]

    samples_5d = np.full((1, 5), np.nan)  # Placeholder for 5D samples
    res_dict = DEFAULT_RES_BUNDLE.copy()
    res_dict.pop("samples", None)  # Remove the samples key to avoid confusion
    try:
        # Get Uncertainty Map Slice
        dp_slice, dm_slice, bias_slice, safe_min_raw, safe_max_raw = _get_uncertainty_slices(flow_angle, uncertainty_map)
        safe_min = safe_min_raw + 0.01
        safe_max = safe_max_raw - 0.01

        if safe_min >= safe_max:
            raise ValueError(f"3D Uncertainty map slice for {tag}-plane is empty or degenerate.")

        # Sanitize Valid Arrays
        mask = np.isfinite(vzb_fit) & np.isfinite(v_trans_fit) & np.isfinite(var_array)
        vzb, v_trans, var_arr = vzb_fit[mask], v_trans_fit[mask], var_array[mask]
        w_rdf = weights_rdf[mask]

        if len(vzb) < 2:
            raise ValueError("Not enough finite data points to run 3D MCMC.")

        bz = block_nvec_avg[2]

        # Empirical Guess
        emp_alpha, slope, intercept, weight = _calculate_empirical_guess(vzb, v_trans, var_arr, bz)
        emp_alpha = np.clip(emp_alpha if np.isfinite(emp_alpha) else 1.0, 0.15, 9.85)
        log_emp_alpha = np.log10(emp_alpha)

        ndim = 3
        pos = np.column_stack([
            np.clip(log_emp_alpha + 1e-2 * np.random.randn(nwalkers), safe_min, safe_max),
            0.0 + 1e-2 * np.random.randn(nwalkers),  # eta
            0.0 + 0.1 * np.random.randn(nwalkers)    # gamma
        ])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_hierarchical_3d_ncp,
                                        args=(vzb, v_trans, bz, uncertainty_map[1], 
                                              dp_slice, dm_slice, bias_slice, var_arr, use_t, use_gmm, w_rdf))
        sampler.run_mcmc(pos, nsteps, progress=False)
        log_samples = sampler.get_chain(discard=burn_in, flat=True)

        ani_list = uncertainty_map[1]
        log_ani_list = np.log10(ani_list)
        log_true = log_samples[:, 0]
        log_lookup = np.clip(log_true, log_ani_list[0], log_ani_list[-1])

        eta_chain      = log_samples[:, 1]
        gamma_chain    = log_samples[:, 2]

        log_eff_chain = reconstruct_effective_chain(log_true, eta_chain, log_lookup, 
                                                    np.log10(ani_list), dp_slice, dm_slice, bias_slice)

        samples_true  = 10**np.clip(log_true, -5.0, 5.0)    # Add in clipping to avoid NaN values in power
        samples_eff   = 10**np.clip(log_eff_chain, -5.0, 5.0) # Add in clipping to avoid NaN values in power

        a_med, a_plus_lin, a_minus_lin, a_plus_log, a_minus_log = _get_percentile_errors(samples_true)
        eff_med, eff_plus, eff_minus, eff_plus_log, eff_minus_log = _get_percentile_errors(samples_eff)
        gam_med, gam_plus, gam_minus, gam_plus_log, gam_minus_log = _get_percentile_errors(gamma_chain)

        # Calculate the reduced chi-squared for the effective fit.        
        dof = max(len(vzb) - 2.0, 1.0)
        denom = max(((bz**2) + eff_med * (1.0 - bz**2)), 1e-12)
        M = ((1.0 - eff_med) * bz) / denom
        sq_res = ((v_trans - (M * vzb + gam_med))**2) / var_arr
        
        chi_sq = np.sum(sq_res) / dof
        w_rob = 5.0 / (4.0 + sq_res)
        chi_sq_robust = np.sum(w_rob * sq_res) / dof

        # 7. Construct 5-Column Samples Array (Pad unmeasured plane with NaNs)
        nan_col = np.full_like(samples_true, np.nan)
        if tag == 'x':
            samples_5d = np.column_stack([samples_true, samples_eff, nan_col, gamma_chain, nan_col])
        else:
            samples_5d = np.column_stack([samples_true, nan_col, samples_eff, nan_col, gamma_chain])

        # 8. Build Results Dictionary
        res_dict.update({
            "T_ani_avg": a_med,
            "T_ani_avg_delta_plus": a_plus_lin,
            "T_ani_avg_delta_minus": a_minus_lin,
            "T_ani_avg_delta_plus_log": a_plus_log,
            "T_ani_avg_delta_minus_log": a_minus_log,
            "chi_sq_joint": chi_sq,
            "chi_sq_joint_robust": chi_sq_robust,
            # "samples": samples_5d
        })

        if tag == 'x':
            res_dict.update({
                "T_ani_chi": eff_med,
                "T_ani_chi_delta_plus": eff_plus,
                "T_ani_chi_delta_minus": eff_minus,
                "T_ani_chi_delta_plus_log": eff_plus_log,
                "T_ani_chi_delta_minus_log": eff_minus_log,
                "gamma_chi": gam_med,
                "gamma_chi_delta_plus": gam_plus,
                "gamma_chi_delta_minus": gam_minus,
                "chi_sq_chi": chi_sq,
                "chi_sq_eff_x": chi_sq,
                "chi_sq_chi_robust": chi_sq_robust,
            })
            var_x_out = var_array
            var_y_out = np.full_like(var_array, np.nan)
        else:
            res_dict.update({
                "T_ani_psi": eff_med,
                "T_ani_psi_delta_plus": eff_plus,
                "T_ani_psi_delta_minus": eff_minus,
                "T_ani_psi_delta_plus_log": eff_plus_log,
                "T_ani_psi_delta_minus_log": eff_minus_log,
                "gamma_psi": gam_med,
                "gamma_psi_delta_plus": gam_plus,
                "gamma_psi_delta_minus": gam_minus,
                "chi_sq_psi": chi_sq,
                "chi_sq_eff_y": chi_sq,
                "chi_sq_psi_robust": chi_sq_robust,
            })
            var_x_out = np.full_like(var_array, np.nan)
            var_y_out = var_array

    except Exception as e:
        print(f"Error during 3D MCMC sampling ({tag}-plane): {e}")
        samples_5d = np.full((1, 5), np.nan)
        var_x_out = var_array if tag == 'x' else np.full_like(var_array, np.nan)
        var_y_out = var_array if tag == 'y' else np.full_like(var_array, np.nan)

    return res_dict, samples_5d, var_x_out, var_y_out



def calc_slope_from_anisotropy(alpha, bz):
    term = 1.0 - alpha
    numerator = term * bz
    denominator = 1.0 - term * (1.0 - bz**2)
    
    # Safely handle both scalar and array inputs to prevent divide-by-zero
    if np.isscalar(denominator):
        if np.abs(denominator) < 1e-12:
            return np.nan
        return numerator / denominator
    else:
        # Array safe division
        M = np.full_like(denominator, np.nan, dtype=float)
        valid = np.abs(denominator) >= 1e-12
        M[valid] = numerator[valid] / denominator[valid]
        return M

### This is the main code block that we are using!
def calculate_spc_anisotropy(ds, uncertainty_map, method='Bayesian_MCMC_NCP', likelihood_type='Gauss',
                              threshold=0.5, no_weights=False, weighted_fit=True, groups=None, flow_correct=False,
                              win_max=False, use_avg_bfield=False, plot_diagnostics=False, pos_id=0):
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
    # print(f"Running SPC Anisotropy Pipeline | Method = {method} | Groups = {"None (Single Events)" if groups is None else "Provided"}")

    # From the input dataset, extract all the key supporting information.
    time = get_data(ds, "epoch")

    peak_az = get_data(ds, "peak_azimuth")      # Peak flow angle values
    peak_el = get_data(ds, "peak_elevation")    
    if flow_correct is False:
        vx = get_data(ds, "vel_x")                  # Derived from vx = vz tan(az)
        vy = get_data(ds, "vel_y")                  # Derived from vy = vz tan(el)
    else:
        try:
            vx = get_data(ds, 'vx_corr')
            vy = get_data(ds, 'vy_corr')
        except KeyError:
            print('Corrected vx and vy are not available. Defaulting to standard approx.')
            vx = get_data(ds, "vel_x")
            vy = get_data(ds, "vel_y")
    
    vz = -get_data(ds, "vel_mid")               # Add the negative to rectify sign
    vz_hi = -get_data(ds, "vel_hi")             
    vz_lo = -get_data(ds, "vel_lo")
    bvec = np.atleast_2d(get_data(ds, "b_vec")) 

    # Extract the plate currents and the reduced distribution function
    ia_curr = get_data(ds, "a_current")
    ib_curr = get_data(ds, "b_current")
    ic_curr = get_data(ds, "c_current")
    id_curr = get_data(ds, "d_current")

    itot = ia_curr + ib_curr + ic_curr + id_curr

    # Extract the standard deviations from the dataset
    ia_std = get_data(ds, "a_current_DELTA_VAR")
    ib_std = get_data(ds, "b_current_DELTA_VAR")
    ic_std = get_data(ds, "c_current_DELTA_VAR")
    id_std = get_data(ds, "d_current_DELTA_VAR")

    # Extract the modulator voltage uncertanties
    mv_lo = get_data(ds, "mv_lo")
    mv_hi = get_data(ds, "mv_hi")
    sig_mv_lo = get_data(ds, "mv_lo_DELTA_VAR")
    sig_mv_hi = get_data(ds, "mv_hi_DELTA_VAR")

    # Get the reduced distribution function
    rdf = get_data(ds, "rdf")

    # Get the flow angles
    azimuth    = get_data(ds, "azimuth")
    elevation  = get_data(ds, "elevation")

    # Define the weights

    # We use the RDF as the primary weighting for calculation the moments 
    # and performing the subsequent fitting.
    weights_rdf = rdf/np.nanmax(rdf, axis=1)[:,None]
    weights     = itot/np.nanmax(itot, axis=1)[:,None]
    mask_weights = get_contiguous_peak_mask(weights, threshold)

    # Use the RDF as the weights for the moment, strictly within the mask
    delta_vz = vz_hi - vz_lo    # Velocity width for the vz measurement
    moment_weights = np.where(mask_weights, rdf, 0.0)
    # Protect against divide by zero!
    integral_weights = np.where(mask_weights, rdf * delta_vz, 0.0)

    sum_weights = np.sum(integral_weights, axis=1)
    sum_weights = np.where(sum_weights == 0, np.nan, sum_weights)

    # Get the u_z moment value!
    u_z = (
        np.sum(
            np.where(mask_weights, vz * integral_weights, 0.0), axis=1
        )  #[cite: 8]
        / sum_weights
    )
    valid_rows = np.any(mask_weights, axis=1)   # Rows with at least one valid weight
    max_idx = np.zeros(rdf.shape[0], dtype=int)
    max_idx[valid_rows] = np.nanargmax(np.where(mask_weights[valid_rows], rdf[valid_rows], np.nan), axis=1)
    row_idx = np.arange(rdf.shape[0])

    u_x = -u_z * np.tan(np.radians(azimuth[row_idx, max_idx]))
    u_y = -u_z * np.tan(-np.radians(elevation[row_idx, max_idx]))

    uvec = np.squeeze(np.dstack([u_x, u_y, u_z]))

    if no_weights:      # This is a very rare flag to pass and is generally not used.
        mask_weights = np.isfinite(weights)
        weights = np.ones_like(weights)

    nvec = bvec / np.linalg.norm(bvec, axis=1)[:,None]

    # Vz variance propagation
    vz_geo_center = (vz_hi + vz_lo) / 2.0
    vz_bias_sq = (vz - vz_geo_center)**2
    vz_quant_var = ((vz_hi - vz_lo)**2) / 12.0
    vz_hw_var = calc_vz_hardware_variance(mv_lo, mv_hi, sig_mv_lo, sig_mv_hi)
    vz_var_tot = vz_quant_var + vz_bias_sq + vz_hw_var

    base_dict = {
        'vx': vx, 'vy': vy, 'vz': vz,
        'rdf': rdf, 'weights': weights, 'weights_rdf': weights_rdf,
        'mask': mask_weights, 'peak_az': peak_az, 'peak_el': peak_el,
        'bvec': bvec, 'nvec': nvec, 'uvec': uvec,
        'itot': itot, 'ia_curr': ia_curr, 'ib_curr': ib_curr,
        'ic_curr': ic_curr, 'id_curr': id_curr,
        'ia_std': ia_std, 'ib_std': ib_std, 'ic_std': ic_std, 'id_std': id_std,
        'vz_var': vz_var_tot, 'time': time,
    }

    static_length = vx.shape[1]
    nan_template = BASE_NAN_TEMPLATE.copy()
    all_results = []

    # Handle Single Events
    if groups is None:
        groups = [(t,t) for t in time]

    range_counter = 0

    for (t0, t1) in tqdm(groups, position=pos_id, desc=f"Core {pos_id}", leave=False):
        # count += 1
        idx_start = np.searchsorted(time, np.datetime64(t0)) 
        idx_end = np.searchsorted(time, np.datetime64(t1)) 

        if t0 == t1:
            idx_end = idx_start + 1

        block_time = t0 + (t1 - t0)/2 

        if idx_start >= idx_end:
            nan_rec = nan_template.copy()
            nan_rec['time'] = block_time
            all_results.append(nan_rec)
            continue

        bvec_slice = base_dict['bvec'][idx_start:idx_end]
        valid_bvec = ~np.all(np.isnan(bvec_slice), axis=1)

        if not np.any(valid_bvec):
            nan_rec = nan_template.copy()
            nan_rec['time'] = block_time
            all_results.append(nan_rec)
            continue

        # Supposedly fast dictonary-based slicing of the block.
        block = {key: val[idx_start:idx_end][valid_bvec] for key, val in base_dict.items()}
        
        # Apply Magnetic Field Geometry
        if use_avg_bfield:
            block_bvec_avg_calc = np.nanmean(block['bvec'], axis=0)
            block_nvec_avg = block_bvec_avg_calc / np.linalg.norm(block_bvec_avg_calc)
            vz_shifted = block['vz'] - block['uvec'][:, 2][:, None]
        else:
            vz_shifted = block['vz'] - block['uvec'][:, 2][:, None]
            block_nvec_avg = np.nanmean(block['nvec'], axis=0)
            block_nvec_avg /= np.linalg.norm(block_nvec_avg)

        block_bvec_to_save = np.nanmean(block['bvec'], axis=0)
        block_bvec_var = np.nanvar(block['bvec'], axis=0)
        b_var_tot = np.nansum(block_bvec_var)

        xangle = np.degrees(np.arctan(block_nvec_avg[0] / block_nvec_avg[2]))
        yangle = np.degrees(np.arctan(block_nvec_avg[1] / block_nvec_avg[2]))

        geom_ok_x = np.abs(xangle) >= 4.0
        geom_ok_y = np.abs(yangle) >= 4.0

        mask_core = block['mask']
        rel_itot = block['weights']

        I_x_pos = block['ia_curr'] + block['id_curr']  # A + D (+x)
        I_x_neg = block['ib_curr'] + block['ic_curr']  # B + C (-x)

        I_y_pos = block['ia_curr'] + block['ib_curr']  # A + B (+y)
        I_y_neg = block['ic_curr'] + block['id_curr']  # C + D (-y)

        # Reject reverse-biased / dead channels
        plates_positive = (
            (block['ia_curr'] > -0.2) & (block['ib_curr'] > -0.2) &
            (block['ic_curr'] > -0.2) & (block['id_curr'] > -0.2)
        )

        itot_floor     = 5.0
        rel_itot_floor = 0.5  # Relative itot floor
        pair_floor     = 5.0  # Current floor in pA for each half-plane pair

        current_ok = (block['itot'] >= itot_floor) & (rel_itot >= rel_itot_floor) & plates_positive
        # current_ok = (block['ia_curr'] >= itot_floor) & (block['ib_curr'] >= itot_floor) & (block['ic_curr'] >= itot_floor) & (block['id_curr'] >= itot_floor) & plates_positive

        # itot_floor = 3.0  # Total current floor in pA for the entire block
        valid_mask_x = mask_core & current_ok & plates_positive & (I_x_pos >= pair_floor) & (I_x_neg >= pair_floor)
        valid_mask_y = mask_core & current_ok & plates_positive & (I_y_pos >= pair_floor) & (I_y_neg >= pair_floor)

        valid_mask_joint = valid_mask_x & valid_mask_y

        n_joint = np.sum(valid_mask_joint)
        n_x_only = np.sum(valid_mask_x)
        n_y_only = np.sum(valid_mask_y)

        n_samples = block['time'].size

        data_vars = nan_template.copy()
        data_vars.update({
            "Nx": block_nvec_avg[0], "Ny": block_nvec_avg[1], "Nz": block_nvec_avg[2],
            "Bx": block_bvec_to_save[0], "By": block_bvec_to_save[1], "Bz": block_bvec_to_save[2],
            "B_var_x": block_bvec_var[0], "B_var_y": block_bvec_var[1], "B_var_z": block_bvec_var[2],
            "B_var_tot": b_var_tot, "B_x_flow_angle": xangle, "B_y_flow_angle": yangle,
            "peak_azimuth": np.nanmean(block['peak_az']).item(),
            "peak_elevation": np.nanmean(block['peak_el']).item(),
            "n_samples": n_samples,
        })

        min_fit_points = 3

        if n_joint >= min_fit_points and geom_ok_x and geom_ok_y and method in ("Bayesian_MCMC_5D_NCP", "Bayesian_MCMC_NCP"):
            # --- 5D JOINT CONSENSUS FIT ---
            valid_idx = valid_mask_joint
            if weighted_fit:
                block['weight_fits'] = block['weights_rdf'][valid_idx]
            else:
                block['weight_fits'] = np.ones(n_joint)

            vx_fit = (block['vx'] - block['uvec'][:, 0][:, None])[valid_idx]
            vy_fit = (block['vy'] - block['uvec'][:, 1][:, None])[valid_idx]
            vz_var = block['vz_var'][valid_idx]

            if use_avg_bfield:
                vzbx_fit = (vz_shifted * block_nvec_avg[0])[valid_idx]
                vzby_fit = (vz_shifted * block_nvec_avg[1])[valid_idx]
            else:
                vzbx_fit = (vz_shifted * block['nvec'][:, 0][:, None])[valid_idx]
                vzby_fit = (vz_shifted * block['nvec'][:, 1][:, None])[valid_idx]

            update_dict, samples, var_x_array, var_y_array = _run_bayesian_pipeline_5d_ncp(
                block, valid_idx, vzbx_fit, vx_fit, vzby_fit, vy_fit, vz_var,
                block_nvec_avg, uncertainty_map, xangle, yangle, static_length,
                likelihood_type=likelihood_type
            )
            data_vars.update(update_dict)

        elif n_x_only >= min_fit_points and geom_ok_x and method in ("Bayesian_MCMC_5D_NCP", "Bayesian_MCMC_NCP"):
            # --- 3D X-ONLY FALLBACK ---
            valid_idx = valid_mask_x
            if weighted_fit:
                block['weight_fits'] = block['weights_rdf'][valid_idx]
            else:
                block['weight_fits'] = np.ones(n_x_only)

            vx_fit = (block['vx'] - block['uvec'][:, 0][:, None])[valid_idx]
            vy_fit = np.array([])  # Empty so plotting function knows Y is unmeasured
            vz_var = block['vz_var'][valid_idx]

            if use_avg_bfield:
                vzbx_fit = (vz_shifted * block_nvec_avg[0])[valid_idx]
                vzby_fit = np.array([])
            else:
                vzbx_fit = (vz_shifted * block['nvec'][:, 0][:, None])[valid_idx]
                vzby_fit = np.array([])

            update_dict, samples, var_x_array, var_y_array = _run_bayesian_pipeline_3d_ncp(
                block, valid_idx, vzbx_fit, vx_fit, vz_var,
                block_nvec_avg, uncertainty_map, xangle, static_length, tag='x',
                likelihood_type=likelihood_type
            )
            data_vars.update(update_dict)

        elif n_y_only >= min_fit_points and geom_ok_y and method in ("Bayesian_MCMC_5D_NCP", "Bayesian_MCMC_NCP"):
            # --- 3D Y-ONLY FALLBACK ---
            valid_idx = valid_mask_y
            if weighted_fit:
                block['weight_fits'] = block['weights_rdf'][valid_idx]
            else:
                block['weight_fits'] = np.ones(n_y_only)

            vx_fit = np.array([])  # Empty so plotting function knows X is unmeasured
            vy_fit = (block['vy'] - block['uvec'][:, 1][:, None])[valid_idx]
            vz_var = block['vz_var'][valid_idx]

            if use_avg_bfield:
                vzbx_fit = np.array([])
                vzby_fit = (vz_shifted * block_nvec_avg[1])[valid_idx]
            else:
                vzbx_fit = np.array([])
                vzby_fit = (vz_shifted * block['nvec'][:, 1][:, None])[valid_idx]

            update_dict, samples, var_x_array, var_y_array = _run_bayesian_pipeline_3d_ncp(
                block, valid_idx, vzby_fit, vy_fit, vz_var,
                block_nvec_avg, uncertainty_map, yangle, static_length, tag='y',
                likelihood_type=likelihood_type
            )
            data_vars.update(update_dict)

        else:
            # Neither plane is statistically or geometrically viable
            nan_rec = nan_template.copy()
            nan_rec['time'] = block_time
            all_results.append(nan_rec)
            continue

        if plot_diagnostics:
            try:
                date_folder = str(block_time)[:10]
                plot_dir = f'test/5D_3/{date_folder}'
                os.makedirs(plot_dir, exist_ok=True)
                safe_time = str(block_time).replace(':', '').replace('-', '').replace('T', '_')
                plot_spc_diagnostic_with_plates(block, valid_idx, block_time, 
                                        vzbx_fit, vx_fit, var_x_array,
                                        vzby_fit, vy_fit, var_y_array,
                                        block_nvec_avg, update_dict,
                                        save_path=f'{plot_dir}/spc_diagnostics_{safe_time}.png')

                # date_folder = str(block_time)[:10] 
                # plot_dir = f'posterior_plots/5D/{date_folder}'
                # os.makedirs(plot_dir, exist_ok=True)
                # safe_time = str(block_time).replace(':', '').replace('-', '').replace('T', '_')
                
                # # Transform only the Alpha columns to log10
                # plot_samples = np.copy(samples)
                # plot_samples[:, 0:3] = np.log10(plot_samples[:, 0:3])
                
                # # Update labels for 5D
                # labels = [r"$\log_{10}(\alpha_{true})$", r"$\log_{10}(\alpha_{eff,x})$", 
                #           r"$\log_{10}(\alpha_{eff,y})$", r"$\gamma_{int,x}$", r"$\gamma_{int,y}$"]
                            
                # fig = corner.corner(plot_samples, labels=labels, 
                #                     quantiles=[0.16, 0.5, 0.84], show_titles=True)
                # plt.suptitle(f"5D MCMC Posterior: {block_time}", y=1.05, fontsize=14)
                # plt.savefig(f'{plot_dir}/5D_corner_{safe_time}.png', dpi=150, bbox_inches='tight')
                # plt.close()
            except Exception as e:
                print(f"Diagnostics plotting failed for {block_time}: {e}")

        data_vars['time'] = block_time
        all_results.append(data_vars)

    # 4. Instantaneous single-pass DataFrame to xarray conversion
    if len(all_results) > 0:
        df = pd.DataFrame(all_results).set_index("time")
        final_ds = df.to_xarray()
    else:
        print("Warning: No valid data blocks found. Returning empty Dataset.")
        final_ds = xr.Dataset()

    return final_ds 
            
