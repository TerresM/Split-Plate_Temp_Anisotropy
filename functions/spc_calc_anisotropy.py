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
from scipy.special import ellipe        # Complete elliptic integral of the second kind!

BASE_NAN_TEMPLATE = {
    "Nx": np.nan, "Ny": np.nan, "Nz": np.nan,
    "Bx": np.nan, "By": np.nan, "Bz": np.nan,
    "B_var_x": np.nan, "B_var_y": np.nan, "B_var_z": np.nan, "B_var_tot": np.nan,
    "B_x_flow_angle": np.nan, "B_y_flow_angle": np.nan,
    "peak_azimuth": np.nan, "peak_elevation": np.nan,
    
    "T_ani_avg": np.nan, "T_ani_avg_uncert": np.nan, 
    "T_ani_avg_delta_plus": np.nan, "T_ani_avg_delta_minus": np.nan,
    
    "T_ani_chi": np.nan, "T_ani_chi_delta_plus": np.nan, "T_ani_chi_delta_minus": np.nan,
    "T_ani_psi": np.nan, "T_ani_psi_delta_plus": np.nan, "T_ani_psi_delta_minus": np.nan,
    
    "gamma_chi": np.nan, "gamma_chi_delta_plus": np.nan, "gamma_chi_delta_minus": np.nan,
    "gamma_psi": np.nan, "gamma_psi_delta_plus": np.nan, "gamma_psi_delta_minus": np.nan, 

    # These values are obsollete
    "slope_chi": np.nan, "r_sq_chi": np.nan,
    "slope_psi": np.nan, "r_sq_psi": np.nan,

    "chi_sq_joint": np.nan, "chi_sq_chi": np.nan, "chi_sq_psi": np.nan,
    "chi_sq_joint_cauchy": np.nan, "chi_sq_chi_cauchy": np.nan, "chi_sq_psi_cauchy": np.nan,
    "chi_sq_eff_x": np.nan, "chi_sq_eff_y": np.nan,
    "gamma_int_x": np.nan, "gamma_int_y": np.nan,

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

def run_emcee_hierarchical_5d_ncp(vzbx, vx, vzby, vy, bz, uncert_map, 
                                  xangle, yangle, var_x_array, var_y_array, 
                                  likelihood_type = "Gauss", 
                                  mix_weights = None):
    use_t = True if likelihood_type.lower() == "t" else False
    use_gmm = True if likelihood_type.lower() == "gmm" else False

    if use_gmm and mix_weights is None:
        raise ValueError("Weights must be provided when using the GMM likelihood method.")

    mask_x = np.isfinite(vzbx) & np.isfinite(vx) & np.isfinite(var_x_array)
    vzbx, vx, var_x_array = vzbx[mask_x], vx[mask_x], var_x_array[mask_x]
    
    mask_y = np.isfinite(vzby) & np.isfinite(vy) & np.isfinite(var_y_array)
    vzby, vy, var_y_array = vzby[mask_y], vy[mask_y], var_y_array[mask_y]

    mix_weights_x = mix_weights[mask_x] if mix_weights is not None else np.zeros_like(var_x_array)
    mix_weights_y = mix_weights[mask_y] if mix_weights is not None else np.zeros_like(var_y_array)

    if len(vzbx) < 2 or len(vzby) < 2:
        raise ValueError("Not enough finite data points to run 5D MCMC.")

    # Calculate Initial Alpha Guesses
    emp_alpha_x, slope_x, intercept_x, weight_x = _calculate_empirical_guess(vzbx, vx, var_x_array, bz)
    emp_alpha_y, slope_y, intercept_y, weight_y = _calculate_empirical_guess(vzby, vy, var_y_array, bz)

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

    # 3. Handle Uncertainty Map Lookup
    try:
        angle_list, ani_list, dp_map, dm_map, bias_map = uncert_map
    # NOTE: this is obsolete once the final code is updated!
    except ValueError:
        angle_list, ani_list, dp_map, dm_map = uncert_map
        bias_map = np.zeros_like(dp_map)

    idx_x = np.nanargmin(np.abs(np.abs(xangle) - angle_list))
    idx_y = np.nanargmin(np.abs(np.abs(yangle) - angle_list))

    dp_slice_x, dm_slice_x, bias_slice_x = dp_map[:, idx_x], dm_map[:, idx_x], bias_map[:, idx_x]
    dp_slice_y, dm_slice_y, bias_slice_y = dp_map[:, idx_y], dm_map[:, idx_y], bias_map[:, idx_y]

    # Find where the X and Y slices actually contain valid data
    valid_x = np.isfinite(dp_slice_x) & np.isfinite(dm_slice_x) & np.isfinite(bias_slice_x)
    valid_y = np.isfinite(dp_slice_y) & np.isfinite(dm_slice_y) & np.isfinite(bias_slice_y)

    log_ani_list = np.log10(ani_list)
    safe_min_x, safe_max_x = log_ani_list[valid_x][0], log_ani_list[valid_x][-1]
    safe_min_y, safe_max_y = log_ani_list[valid_y][0], log_ani_list[valid_y][-1]

    # Use the most restrictive overlapping boundary between X and Y
    safe_min = max(safe_min_x, safe_min_y) + 0.01
    safe_max = min(safe_max_x, safe_max_y) - 0.01

    # 4. Initialize the 5D Walkers
    nwalkers = 16
    ndim = 5        

    # Initialize the alpha true walkers as a random variable number added to the empirical guess.
    log_pos_alpha_true  = log_emp_alpha + 1e-4 * np.random.randn(nwalkers)
    log_pos_alpha_true = np.clip(log_pos_alpha_true, safe_min, safe_max)

    # Initialize the eta parameters. For this we will start at zero!
    pos_eta_x = 0.0 + 1e-4 * np.random.randn(nwalkers)
    pos_eta_y = 0.0 + 1e-4 * np.random.randn(nwalkers)

    # Define the initial intercept values
    pos_gamma_x = gamma_guess_x + 0.1 * np.random.randn(nwalkers)
    pos_gamma_y = gamma_guess_y + 0.1 * np.random.randn(nwalkers)

    # Our initial parameter space!
    pos = np.vstack((log_pos_alpha_true, pos_eta_x, pos_eta_y, pos_gamma_x, pos_gamma_y)).T

    
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_hierarchical_5d_ncp,
                                        args = (vzbx, vx, vzby, vy, bz, ani_list, 
                                        dp_slice_x, dm_slice_x, bias_slice_x, 
                                        dp_slice_y, dm_slice_y, bias_slice_y, 
                                        var_x_array, var_y_array, use_t, use_gmm, 
                                        mix_weights_x, mix_weights_y) 
                )

        sampler.run_mcmc(pos, 400, progress=False)

        log_samples = sampler.get_chain(discard=50, flat=True)

        log_true_chain = log_samples[:, 0]
        eta_x_chain    = log_samples[:, 1]
        eta_y_chain    = log_samples[:, 2]
        gamma_x_chain  = log_samples[:, 3]
        gamma_y_chain  = log_samples[:, 4]

        log_ani_list = np.log10(ani_list)
        log_true_lookup = np.clip(log_true_chain, log_ani_list[0], log_ani_list[-1])

        log_eff_x_chain = reconstruct_effective_chain(log_true_chain, eta_x_chain, log_true_lookup, 
                                                      log_ani_list, dp_slice_x, dm_slice_x, bias_slice_x)
        
        log_eff_y_chain = reconstruct_effective_chain(log_true_chain, eta_y_chain, log_true_lookup, 
                                                      log_ani_list, dp_slice_y, dm_slice_y, bias_slice_y)
        
        samples_true  = 10**(log_true_chain)
        samples_eff_x = 10**(log_eff_x_chain)
        samples_eff_y = 10**(log_eff_y_chain)

        samples = np.vstack((samples_true, samples_eff_x, samples_eff_y, gamma_x_chain, gamma_y_chain)).T
        
        alpha_16, alpha_50, alpha_84 = np.percentile(samples_true, [16, 50, 84])
        alpha_eff_x_16, alpha_eff_x_50, alpha_eff_x_84 = np.percentile(samples_eff_x, [16, 50, 84])
        alpha_eff_y_16, alpha_eff_y_50, alpha_eff_y_84 = np.percentile(samples_eff_y, [16, 50, 84])

        gamma_eff_x_16, gamma_eff_x_50, gamma_eff_x_84 = np.percentile(gamma_x_chain, [16, 50, 84])
        gamma_eff_y_16, gamma_eff_y_50, gamma_eff_y_84 = np.percentile(gamma_y_chain, [16, 50, 84])

        res_bundle = {
            'alpha'           : alpha_50,
            'sig_plus'        : alpha_84 - alpha_50,
            'sig_minus'       : alpha_50 - alpha_16,
            'alpha_eff_x'     : alpha_eff_x_50,
            'sig_plus_x'      : alpha_eff_x_84 - alpha_eff_x_50,
            'sig_minus_x'     : alpha_eff_x_50 - alpha_eff_x_16,
            'alpha_eff_y'     : alpha_eff_y_50,
            'sig_plus_y'      : alpha_eff_y_84 - alpha_eff_y_50,
            'sig_minus_y'     : alpha_eff_y_50 - alpha_eff_y_16,
            'gamma_x'         : gamma_eff_x_50,
            'sig_plus_gam_x'  : gamma_eff_x_84 - gamma_eff_x_50,
            'sig_minus_gam_x' : gamma_eff_x_50 - gamma_eff_x_16,
            'gamma_y'         : gamma_eff_y_50,
            'sig_plus_gam_y'  : gamma_eff_y_84 - gamma_eff_y_50,
            'sig_minus_gam_y' : gamma_eff_y_50 - gamma_eff_y_16,
            'samples'         : samples 
        }

    except Exception as e:
        # If emcee fails completely, fail gracefully by returning NaNs.
        # This allows the rest of the loop to process chi-squared logic natively as NaN
        res_bundle = {
            'alpha'           : np.nan, 'sig_plus'        : np.nan, 'sig_minus'       : np.nan,
            'alpha_eff_x'     : np.nan, 'sig_plus_x'      : np.nan, 'sig_minus_x'     : np.nan,
            'alpha_eff_y'     : np.nan, 'sig_plus_y'      : np.nan, 'sig_minus_y'     : np.nan,
            'gamma_x'         : np.nan, 'sig_plus_gam_x'  : np.nan, 'sig_minus_gam_x' : np.nan,
            'gamma_y'         : np.nan, 'sig_plus_gam_y'  : np.nan, 'sig_minus_gam_y' : np.nan,
            'samples'         : np.full((1, 5), np.nan) 
        }

    return res_bundle

def _run_bayesian_pipeline_5d_ncp(block, valid_idx, vzbx_fit, vx_fit, 
                              vzby_fit, vy_fit, vz_var, block_nvec_avg,
                              uncertainty_map, xangle, yangle, static_length, 
                              likelihood_type='Gauss'):

    lamda = 1.009       
    C = lamda/2.0

    # Grab in the plate currents that will be used for defining the plate differences
    ia_fit = block.ia_curr.values[valid_idx]
    ib_fit = block.ib_curr.values[valid_idx]
    ic_fit = block.ic_curr.values[valid_idx]
    id_fit = block.id_curr.values[valid_idx]

    i_tot = np.maximum(ia_fit + ib_fit + ic_fit + id_fit, 1e-6)     
    i_diff_x = (ia_fit + id_fit) - (ib_fit + ic_fit)
    i_diff_y = (ic_fit + id_fit) - (ia_fit + ib_fit)

    # Convert the standard deviations to variances
    var_A = block.ia_std.values[valid_idx]**2
    var_B = block.ib_std.values[valid_idx]**2
    var_C = block.ic_std.values[valid_idx]**2
    var_D = block.id_std.values[valid_idx]**2

    # X-Z Plane: Positive = A, D | Negative = B, C
    Sigma_pos_x_sq = var_A + var_D
    Sigma_neg_x_sq = var_B + var_C

    # Y-Z Plane: Positive = A, B | Negative = C, D
    Sigma_pos_y_sq = var_A + var_B
    Sigma_neg_y_sq = var_C + var_D

    # Get the raw vx and vy values
    vx_raw = block.vx.values[valid_idx]
    vy_raw = block.vy.values[valid_idx]
    vz_raw = block.vz.values[valid_idx] 
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

    norm_weights = i_tot / np.sum(i_tot)
    
    # The variance of the bulk velocity is the sum of the 
    # squared normalized weights * the measurement variances
    term3_x = np.sum((norm_weights**2) * term2_x)
    term3_y = np.sum((norm_weights**2) * term2_y)

    var_x_array_inst = np.maximum(term2_x, 1e-4) # + term3_x, 1e-4)
    var_y_array_inst = np.maximum(term2_y, 1e-4) # + term3_y, 1e-4)

    safe_weights = np.maximum(block.weight_fits.values, 1e-4)

    var_x_array = var_x_array_inst / safe_weights
    var_y_array = var_y_array_inst / safe_weights

    weights_rdf = block.weights_rdf.values[valid_idx]

    # Call the NCP EMCEE sampler
    res_bundle = run_emcee_hierarchical_5d_ncp(
        vzbx_fit, vx_fit, vzby_fit, vy_fit, block_nvec_avg[2], 
        uncertainty_map, xangle, yangle, var_x_array, var_y_array, 
        likelihood_type, mix_weights=weights_rdf
    )

    # --- FIXED COPY-PASTE BUGS HERE ---
    avg_alpha = get_data(res_bundle, 'alpha')
    avg_sig_plus = get_data(res_bundle, 'sig_plus')
    avg_sig_minus = get_data(res_bundle, 'sig_minus')

    alpha_eff_x  = get_data(res_bundle, 'alpha_eff_x')
    sig_plus_x   = get_data(res_bundle, 'sig_plus_x')
    sig_minus_x  = get_data(res_bundle, 'sig_minus_x') # Fixed

    alpha_eff_y  = get_data(res_bundle, 'alpha_eff_y')
    sig_plus_y   = get_data(res_bundle, 'sig_plus_y')
    sig_minus_y  = get_data(res_bundle, 'sig_minus_y') # Fixed

    gamma_x = get_data(res_bundle, 'gamma_x')
    sig_plus_gam_x = get_data(res_bundle, 'sig_plus_gam_x' )
    sig_minus_gam_x = get_data(res_bundle, 'sig_minus_gam_x')

    gamma_y = get_data(res_bundle, 'gamma_y')
    sig_plus_gam_y  = get_data(res_bundle, 'sig_plus_gam_y' ) # Fixed
    sig_minus_gam_y = get_data(res_bundle, 'sig_minus_gam_y') # Fixed

    samples = get_data(res_bundle, 'samples')

    bz = block_nvec_avg[2]
    
    # Get the effective x and y fits.
    m_x = ((1.0 - avg_alpha) * bz) / (1.0 - (1.0 - avg_alpha) * (1.0 - bz**2))
    m_y = ((1.0 - avg_alpha) * bz) / (1.0 - (1.0 - avg_alpha) * (1.0 - bz**2))

    m_eff_x = ((1.0 - alpha_eff_x) * bz) / (1.0 - (1.0 - alpha_eff_x) * (1.0 - bz**2))
    m_eff_y = ((1.0 - alpha_eff_y) * bz) / (1.0 - (1.0 - alpha_eff_y) * (1.0 - bz**2))
    
    sq_res_x_valid = ((vx_fit - (m_x * vzbx_fit + gamma_x))**2)/ var_x_array
    sq_res_y_valid = ((vy_fit - (m_y * vzby_fit + gamma_y))**2)/ var_y_array

    sq_res_eff_x_valid = ((vx_fit - (m_eff_x * vzbx_fit + gamma_x))**2)/ var_x_array
    sq_res_eff_y_valid = ((vy_fit - (m_eff_y * vzby_fit + gamma_y))**2)/ var_y_array

    dof_single = max(len(vzbx_fit) - 2.0, 1.0)
    dof_joint = max((2.0 * len(vzbx_fit)) - 3.0, 1.0)

    chi_sq_x = np.sum(sq_res_x_valid) / dof_single
    chi_sq_y = np.sum(sq_res_y_valid) / dof_single
    chi_sq_joint = (np.sum(sq_res_x_valid) + np.sum(sq_res_y_valid)) / dof_joint

    # Cauchy Pseudo-Chi-Squared Calculations
    chi_sq_x_cauchy = np.sum(2.0 * np.log(1.0 + 0.5 * sq_res_x_valid)) / dof_single
    chi_sq_y_cauchy = np.sum(2.0 * np.log(1.0 + 0.5 * sq_res_y_valid)) / dof_single
    chi_sq_joint_cauchy = (dof_single * chi_sq_x_cauchy + dof_single * chi_sq_y_cauchy) / dof_joint

    chi_sq_eff_x = np.sum(sq_res_eff_x_valid) / dof_single
    chi_sq_eff_y = np.sum(sq_res_eff_y_valid) / dof_single

    # Create an emply 2D array shaped exactly like the input block (time, dim)
    block_shape = valid_idx.shape
    block_res_x = np.full(block_shape, np.nan)
    block_res_y = np.full(block_shape, np.nan)
    
    # Map the 1D valid residuals back into their exact physical velocity bins
    block_res_x[valid_idx] = sq_res_x_valid
    block_res_y[valid_idx] = sq_res_y_valid 

    # --- FIXED DICT KEYS HERE ---
    update_dict = {
        "T_ani_avg": avg_alpha,
        "T_ani_avg_delta_plus": avg_sig_plus,
        "T_ani_avg_delta_minus": avg_sig_minus,
        "T_ani_chi" : alpha_eff_x,
        "T_ani_chi_delta_plus" : sig_plus_x,
        "T_ani_chi_delta_minus" : sig_minus_x,
        "T_ani_psi" : alpha_eff_y,
        "T_ani_psi_delta_plus" : sig_plus_y,
        "T_ani_psi_delta_minus" : sig_minus_y,
        "gamma_chi" : gamma_x,
        "gamma_chi_delta_plus" : sig_plus_gam_x,
        "gamma_chi_delta_minus" : sig_minus_gam_x,
        "gamma_psi" : gamma_y,
        "gamma_psi_delta_plus" : sig_plus_gam_y,   # Fixed
        "gamma_psi_delta_minus" : sig_minus_gam_y, # Fixed
        "chi_sq_joint": chi_sq_joint,
        "chi_sq_chi": chi_sq_x,
        "chi_sq_psi": chi_sq_y,
        "chi_sq_eff_x": chi_sq_eff_x,
        "chi_sq_eff_y": chi_sq_eff_y,
        "chi_sq_chi_cauchy": chi_sq_x_cauchy,
        "chi_sq_psi_cauchy": chi_sq_y_cauchy,
        "chi_sq_joint_cauchy": chi_sq_joint_cauchy,
    }

    # We also return samples and var arrays so the main loop can handle plotting!
    return update_dict, samples, var_x_array, var_y_array

@njit(cache=True, nogil=True)
def log_prior_hierarchical_3d_ncp(theta):
    log_alpha_true, eta, gamma_int = theta

    if not (-2.0 < log_alpha_true < 2.0): return -np.inf
    if not (-100 < gamma_int < 100): return -np.inf

    log_P_eta = -0.5 * eta**2
    log_P_gamma = -0.5 * (gamma_int / 10.0)**2

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
                                  static_length, tag='x', likelihood_type='Gauss'):
    
    use_t = True if likelihood_type.lower() == 't' else False
    use_gmm = True if likelihood_type.lower() == 'gmm' else False

    lamda = 1.009
    # 1. Calculate the single-plane variance (simplified from 5D)
    # Grab in the plate currents that will be used for defining the plate differences
    ia_fit = block.ia_curr.values[valid_idx]
    ib_fit = block.ib_curr.values[valid_idx]
    ic_fit = block.ic_curr.values[valid_idx]
    id_fit = block.id_curr.values[valid_idx]

    i_tot = np.maximum(ia_fit + ib_fit + ic_fit + id_fit, 1e-6)     
    i_diff_x = (ia_fit + id_fit) - (ib_fit + ic_fit)
    i_diff_y = (ic_fit + id_fit) - (ia_fit + ib_fit)

    # Convert the standard deviations to variances
    var_A = block.ia_std.values[valid_idx]**2
    var_B = block.ib_std.values[valid_idx]**2
    var_C = block.ic_std.values[valid_idx]**2
    var_D = block.id_std.values[valid_idx]**2

    # X-Z Plane: Positive = A, D | Negative = B, C
    Sigma_pos_x_sq = var_A + var_D
    Sigma_neg_x_sq = var_B + var_C

    # Y-Z Plane: Positive = A, B | Negative = C, D
    Sigma_pos_y_sq = var_A + var_B
    Sigma_neg_y_sq = var_C + var_D
    
    # Get the raw vx and vy values
    vx_raw = block.vx.values[valid_idx]
    vy_raw = block.vy.values[valid_idx]
    vz_raw = block.vz.values[valid_idx] 
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

    safe_weights = np.maximum(block.weight_fits.values, 1e-4)**3

    weights_rdf = block.weights_rdf.values[valid_idx]

    if tag == 'x':
        var_array = np.maximum(term2_x, 1e-4) / safe_weights
    else:
        var_array = np.maximum(term2_y, 1e-4) / safe_weights
        
    # 2. Get Uncertainty Map Slice
    angle_list, ani_list, dp_map, dm_map, bias_map = uncertainty_map
    idx_angle = np.nanargmin(np.abs(np.abs(flow_angle) - angle_list))
    dp_slice = dp_map[:, idx_angle]
    dm_slice = dm_map[:, idx_angle]
    bias_slice = bias_map[:, idx_angle]

    bz = block_nvec_avg[2]

    mask = np.isfinite(vzb_fit) & np.isfinite(v_trans_fit) & np.isfinite(var_array)
    vzb_fit, v_trans_fit, var_array = vzb_fit[mask], v_trans_fit[mask], var_array[mask]
    weights_rdf = weights_rdf[mask]
    if len(vzb_fit) < 2:
        raise ValueError("Not enough finite data points to run 3D MCMC.")
    
    # 3. Empirical Guess
    emp_alpha, slope, intercept, weight = _calculate_empirical_guess(vzb_fit, v_trans_fit, var_array, bz)
    emp_alpha = np.clip(emp_alpha if np.isfinite(emp_alpha) else 1.0, 0.15, 9.85)
    log_emp_alpha = np.log10(emp_alpha)
    gamma_guess = 0.0 #intercept if np.isfinite(intercept) else 0.0

    # 4. Initialize 3D Walkers
    valid = np.isfinite(dp_slice) & np.isfinite(dm_slice) & np.isfinite(bias_slice)
    
    # Check if the slice is completely empty
    if np.sum(valid) < 2:
        raise ValueError("3D Uncertainty map slice is empty.")
        
    log_ani_list = np.log10(ani_list)
    safe_min = log_ani_list[valid][0] + 0.01
    safe_max = log_ani_list[valid][-1] - 0.01

    nwalkers, ndim = 16, 3
    log_pos_alpha_true = log_emp_alpha + 1e-1 * np.random.randn(nwalkers)
    
    # Dynamically clip the 3D walkers!
    log_pos_alpha_true = np.clip(log_pos_alpha_true, safe_min, safe_max)

    pos = np.vstack((
        log_pos_alpha_true,
        0.0 + 1e-1 * np.random.randn(nwalkers),
        gamma_guess + 2.0 * np.random.randn(nwalkers)
    )).T

    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_hierarchical_3d_ncp,
                                        args=(vzb_fit, v_trans_fit, bz, ani_list, 
                                              dp_slice, dm_slice, bias_slice, var_array, use_t, use_gmm, weights_rdf))
        sampler.run_mcmc(pos, 400, progress=False)
        log_samples = sampler.get_chain(discard=50, flat=True)

        log_true_chain = log_samples[:, 0]
        eta_chain      = log_samples[:, 1]
        gamma_chain    = log_samples[:, 2]

        log_true_lookup = np.clip(log_true_chain, np.log10(ani_list[0]), np.log10(ani_list[-1]))
        log_eff_chain = reconstruct_effective_chain(log_true_chain, eta_chain, log_true_lookup, 
                                                    np.log10(ani_list), dp_slice, dm_slice, bias_slice)
        
        alpha_50 = np.percentile(10**log_true_chain, 50)
        sig_plus = np.percentile(10**log_true_chain, 84) - alpha_50
        sig_minus = alpha_50 - np.percentile(10**log_true_chain, 16)
        
        alpha_eff_50 = np.percentile(10**log_eff_chain, 50)
        sig_plus_eff = np.percentile(10**log_eff_chain, 84) - alpha_eff_50
        sig_minus_eff = alpha_eff_50 - np.percentile(10**log_eff_chain, 16)
        
        gamma_50 = np.percentile(gamma_chain, 50)
        sig_plus_gam = np.percentile(gamma_chain, 84) - gamma_50
        sig_minus_gam = gamma_50 - np.percentile(gamma_chain, 16)
        
        # Calculate chi_sq for the valid plane
        M = ((1.0 - alpha_eff_50) * bz) / max(((bz**2) + alpha_eff_50 * (1.0 - bz**2)), 1e-30)
        chi_sq_eff = np.sum(((v_trans_fit - (M * vzb_fit + gamma_50))**2) / var_array) / max(len(vzb_fit) - 2.0, 1.0)
        
    except Exception as e:
        alpha_50 = sig_plus = sig_minus = alpha_eff_50 = sig_plus_eff = sig_minus_eff = np.nan
        gamma_50 = sig_plus_gam = sig_minus_gam = chi_sq_eff = np.nan

    # 5. Smart Dictionary Padding
    update_dict = {
        "T_ani_avg": alpha_50,
        "T_ani_avg_delta_plus": sig_plus,
        "T_ani_avg_delta_minus": sig_minus,
        "chi_sq_joint": chi_sq_eff, # For a single plane, joint is just the single plane
    }

    if tag == 'x':
        update_dict.update({
            "T_ani_chi": alpha_eff_50, "T_ani_chi_delta_plus": sig_plus_eff, "T_ani_chi_delta_minus": sig_minus_eff,
            "T_ani_psi": np.nan, "T_ani_psi_delta_plus": np.nan, "T_ani_psi_delta_minus": np.nan,
            "gamma_chi": gamma_50, "gamma_chi_delta_plus": sig_plus_gam, "gamma_chi_delta_minus": sig_minus_gam,
            "gamma_psi": np.nan, "gamma_psi_delta_plus": np.nan, "gamma_psi_delta_minus": np.nan,
            "chi_sq_chi": chi_sq_eff, "chi_sq_psi": np.nan,
            "chi_sq_eff_x": chi_sq_eff, "chi_sq_eff_y": np.nan
        })
        return update_dict, None, var_array, np.full_like(var_array, np.nan)
    else:
        update_dict.update({
            "T_ani_chi": np.nan, "T_ani_chi_delta_plus": np.nan, "T_ani_chi_delta_minus": np.nan,
            "T_ani_psi": alpha_eff_50, "T_ani_psi_delta_plus": sig_plus_eff, "T_ani_psi_delta_minus": sig_minus_eff,
            "gamma_chi": np.nan, "gamma_chi_delta_plus": np.nan, "gamma_chi_delta_minus": np.nan,
            "gamma_psi": gamma_50, "gamma_psi_delta_plus": sig_plus_gam, "gamma_psi_delta_minus": sig_minus_gam,
            "chi_sq_chi": np.nan, "chi_sq_psi": chi_sq_eff,
            "chi_sq_eff_x": np.nan, "chi_sq_eff_y": chi_sq_eff
        })
        return update_dict, None, np.full_like(var_array, np.nan), var_array


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
    weights_rdf = rdf/np.nanmax(rdf, axis=1)[:,None]
    weights = itot/np.nanmax(itot, axis=1)[:,None]
    mask_weights = get_contiguous_peak_mask(weights, threshold)

    # Use the RDF as the weights for the moment, strictly within the mask
    moment_weights = np.where(mask_weights, rdf, 0.0)

    # Sum the weights across the velocity axis (axis=1)
    sum_weights = np.sum(moment_weights, axis=1)

    # Protect against divide by zero!
    sum_weights = np.where(sum_weights == 0, np.nan, sum_weights)

    # Get the u_z moment value!
    u_z = np.sum(np.where(mask_weights, vz, 0.0) * moment_weights, axis=1) / sum_weights
    
    # Get the peak values from the vdf
    max_idx = np.nanargmax(np.where(mask_weights, rdf, np.nan), axis=1) # get the peak value for every time.
    n_times = rdf.shape[0]      # Length of time
    row_idx = np.arange(n_times)

    u_x = -u_z * np.tan(np.radians(azimuth[row_idx, max_idx]))
    u_y = -u_z * np.tan(-np.radians(elevation[row_idx, max_idx]))
    # u_z = np.where(mask_weights, vz, np.nan)[row_idx, max_idx]

    uvec = np.squeeze(np.dstack([u_x, u_y, u_z]))

    if no_weights:      # This is a very rare flag to pass and is generally not used.
        mask_weights = np.isfinite(weights)
        weights = np.ones_like(weights)

    # Package into Dataset for easy block slicing
    base_dict = dict(
        vx=(["time", "dim"], vx),
        vy=(["time", "dim"], vy),
        vz=(["time", "dim"], vz),
        rdf=(["time", "dim"], rdf),
        weights=(["time", "dim"], weights),
        weights_rdf = (["time", "dim"], weights_rdf),
        mask=(["time", "dim"], mask_weights),
        peak_az=("time", peak_az),
        peak_el=("time", peak_el),
        bvec=(["time", "vec"], bvec),
        ia_curr=(["time", "dim"], ia_curr),
        ib_curr=(["time", "dim"], ib_curr),
        ic_curr=(["time", "dim"], ic_curr),
        id_curr=(["time", "dim"], id_curr),
        ia_std=(["time", "dim"], ia_std),
        ib_std=(["time", "dim"], ib_std),
        ic_std=(["time", "dim"], ic_std),
        id_std=(["time", "dim"], id_std),
    )

    # # Pre-calculate the magnetic field geometry if not averaging over the block
    # if not use_avg_bfield:
    nvec = bvec / np.linalg.norm(bvec, axis=1)[:,None]
    base_dict['nvec'] = (["time", "vec"], nvec)
    base_dict['uvec'] = (["time", "vec"], uvec)
    base_dict['itot'] = (["time", "dim"], itot)

    # Vz Error estimation, this comes in for uncertainty calculations
    vz_geo_center = (vz_hi + vz_lo)/ 2.0
    vz_bias_sq = (vz - vz_geo_center)**2
    vz_quant_var = ((vz_hi - vz_lo)**2) / 12.0
    
    # --- MISSING CODE: Call the finite difference function ---
    vz_hw_var = calc_vz_hardware_variance(mv_lo, mv_hi, sig_mv_lo, sig_mv_hi)
    
    # Combine quantization, bias, and hardware jitter
    base_dict['vz_var'] = (["time", "dim"], vz_quant_var + vz_bias_sq + vz_hw_var)

    base = xr.Dataset(base_dict, coords = dict(time=time))
    static_length = base.vx.shape[1]

    nan_template = BASE_NAN_TEMPLATE.copy()

    all_results = []

    # Handle Single Events
    if groups is None:
        groups = [(t,t) for t in time.values]

    count = 0
    raw_time = base.time.values

    for (t0, t1) in tqdm(groups, position=pos_id, desc=f"Core {pos_id}", leave=False):
        count += 1

        # idx_start = np.searchsorted(raw_time, np.datetime64(t0))
        # idx_end = np.searchsorted(raw_time, np.datetime64(t1))

        # if idx_start != idx_end:
        #     block = base.isel(time=slice(idx_start, idx_end))
        # else:
        #     block = base.isel(time=slice(idx_start, idx_start+1))
        idx_start = np.searchsorted(raw_time, np.datetime64(t0)) #
        idx_end = np.searchsorted(raw_time, np.datetime64(t1)) #[cite: 2]

        if t0 == t1:
            # Single event mode: grab the specific timestamp
            block = base.isel(time=slice(idx_start, idx_start+1))
        else:
            # Grouped mode: grab the data between t0 and t1
            # If there is a data gap (idx_start == idx_end), this naturally returns an empty block!
            block = base.isel(time=slice(idx_start, idx_end))

        block_time = t0 + (t1 - t0)/2 

        # Filter out the NaN B-fields
        block_bvec_nan = np.all(np.isnan(block.bvec.data), axis=1)
        if np.all(block_bvec_nan) or block.time.size == 0:
            result_ds = package_to_xarray(nan_template.copy(), block_time, t0, t1)
            all_results.append(result_ds)
            continue
        elif np.any(block_bvec_nan):
            block = block.isel(time=~block_bvec_nan)

        n_samples = block.time.size

        if win_max:
            # weights_block = block.rdf.values / np.nanmax(block.rdf.values)
            weights_block = block.itot.values / np.nanmax(block.itot.values)
            valid_idx = get_contiguous_peak_mask(weights_block, threshold)
            base_weights = weights_block[valid_idx]
        else:
            valid_idx = block.mask.values
            base_weights = block.weights.values[valid_idx]

        # Windowing and weighting masks
        if weighted_fit:
            block['weight_fits'] = base_weights
        else:
            block['weight_fits'] = np.ones(len(block.weights.values[valid_idx]))

        vx_fit = (block.vx.values - block.uvec.values[:,0][:,None])[valid_idx] 
        vy_fit = (block.vy.values - block.uvec.values[:,1][:,None])[valid_idx] 
        vz_var = block.vz_var.values[valid_idx]

        # Apply Magnetic Field Geometry
        if use_avg_bfield:
            block_bvec_avg_calc = np.nanmean(block.bvec.values, axis=0)
            block_nvec_avg = block_bvec_avg_calc / np.linalg.norm(block_bvec_avg_calc)
            vz_shifted = block.vz.values - block.uvec.values[:,2][:,None]
            vzbx_fit = (vz_shifted * block_nvec_avg[0])[valid_idx]
            vzby_fit = (vz_shifted * block_nvec_avg[1])[valid_idx]
        else:
            vz_shifted = block.vz.values - block.uvec.values[:,2][:,None]

            vzbx_fit = (vz_shifted * block.nvec.values[:, 0][:, None])[valid_idx] 
            vzby_fit = (vz_shifted * block.nvec.values[:, 1][:, None])[valid_idx] 
            block_nvec_avg = np.nanmean(block.nvec.values, axis=0)
            block_nvec_avg /= np.linalg.norm(block_nvec_avg)
            

        if len(vzbx_fit) < 2 or len(vzby_fit) < 2:
            result_ds = package_to_xarray(nan_template.copy(), block_time, t0, t1)
            all_results.append(result_ds)
            continue

        block_bvec_to_save = np.nanmean(block.bvec.values, axis=0)

        block_bvec_var = np.nanvar(block.bvec.values, axis=0)
        b_var_tot = np.nansum(block_bvec_var)

        xangle = np.degrees(np.arctan(block_nvec_avg[0] / block_nvec_avg[2]))
        yangle = np.degrees(np.arctan(block_nvec_avg[1] / block_nvec_avg[2]))

        x_only, y_only = False, False
        if np.abs(xangle) < 4: y_only = True
        if np.abs(yangle) < 4: x_only = True

        if x_only and y_only:
            result_ds = package_to_xarray(nan_template.copy(), block_time, t0, t1)
            all_results.append(result_ds)
            continue

        data_vars = nan_template.copy()
        data_vars.update({
            "Nx": block_nvec_avg[0],
            "Ny": block_nvec_avg[1],
            "Nz": block_nvec_avg[2],
            "Bx": block_bvec_to_save[0],
            "By": block_bvec_to_save[1],
            "Bz": block_bvec_to_save[2],
            "B_var_x": block_bvec_var[0],
            "B_var_y": block_bvec_var[1],
            "B_var_z": block_bvec_var[2],
            "B_var_tot": b_var_tot,
            "B_x_flow_angle": xangle,
            "B_y_flow_angle": yangle,
            "peak_azimuth": block.peak_az.mean().item(),
            "peak_elevation": block.peak_el.mean().item(),
            "n_samples": n_samples,
        })

        ia_fit = block.ia_curr.values[valid_idx]
        ib_fit = block.ib_curr.values[valid_idx]
        ic_fit = block.ic_curr.values[valid_idx]
        id_fit = block.id_curr.values[valid_idx]

        i_tot = np.maximum(ia_fit + ib_fit + ic_fit + id_fit, 1e-6)  

        if method == "Bayesian_MCMC_5D_NCP":
            if x_only:
                update_dict, samples, var_x_array, var_y_array = _run_bayesian_pipeline_3d_ncp(
                    block, valid_idx, vzbx_fit, vx_fit, vz_var, 
                    block_nvec_avg, uncertainty_map, xangle, static_length, tag='x',
                    likelihood_type=likelihood_type
                )
            elif y_only:
                update_dict, samples, var_x_array, var_y_array = _run_bayesian_pipeline_3d_ncp(
                    block, valid_idx, vzby_fit, vy_fit, vz_var, 
                    block_nvec_avg, uncertainty_map, yangle, static_length, tag='y',
                    likelihood_type=likelihood_type
                )
            else:
                update_dict, samples, var_x_array, var_y_array = _run_bayesian_pipeline_5d_ncp(
                    block, valid_idx, vzbx_fit, vx_fit, vzby_fit, vy_fit, vz_var, 
                    block_nvec_avg, uncertainty_map, xangle, yangle, static_length, 
                    likelihood_type=likelihood_type
                )

            data_vars.update(update_dict)


            # if block_time == np.datetime64('2023-12-27T03:47:30.000000000'):
            if block_time == np.datetime64('2023-09-26T03:16:58.000000000'):
                bz = block_nvec_avg[2]
                a_x = update_dict['T_ani_chi']
                a_y = update_dict['T_ani_psi']
                aavg = update_dict['T_ani_avg']
                M_x = calc_slope_from_anisotropy(a_x, bz)
                M_y = calc_slope_from_anisotropy(a_y, bz)
                M   = calc_slope_from_anisotropy(aavg, bz)
                
                gam_x = update_dict['gamma_chi']
                gam_y = update_dict['gamma_psi']

                W_x = 1.0 / var_x_array
                cx_x = np.sum(W_x * vzbx_fit) / np.sum(W_x)
                cy_x = np.sum(W_x * vx_fit) / np.sum(W_x)
                gam_true_x = cy_x - (M * cx_x)  # Anchor M through the X-data centroid

                W_y = 1.0 / var_y_array
                cx_y = np.sum(W_y * vzby_fit) / np.sum(W_y)
                cy_y = np.sum(W_y * vy_fit) / np.sum(W_y)
                gam_true_y = cy_y - (M * cx_y)



                fig, ax = plt.subplots(2, figsize=(12,8))

                ax[0].errorbar(vzbx_fit, vx_fit, yerr=np.sqrt(var_x_array), linestyle='None', fmt='o', color='k', ecolor='tab:blue', capsize=5)
                im0 = ax[0].scatter(vzbx_fit, vx_fit, c=i_tot, zorder=10)

                xx = np.linspace(np.min(vzbx_fit), np.max(vzbx_fit))
                ax[0].plot(xx, M_x*xx + gam_x)
                ax[0].plot(xx, M*xx + gam_true_x)

                ax[1].errorbar(vzby_fit, vy_fit, yerr=np.sqrt(var_y_array), linestyle='None', fmt='o', color='k', ecolor='tab:blue', capsize=5)
                im1 = ax[1].scatter(vzby_fit, vy_fit, c=i_tot, zorder=10)
                yy = np.linspace(np.min(vzby_fit), np.max(vzby_fit))
                ax[1].plot(yy, M_y*yy + gam_y)
                ax[1].plot(yy, M*yy + gam_true_y)

                ax[0].set_title(f'{t0} to {t1} | {a_x}, {a_y}, {aavg}')

                fig.colorbar(im0)
                fig.colorbar(im1)


            if plot_diagnostics:
                try:
                    date_folder = str(block_time)[:10] 
                    plot_dir = f'posterior_plots/5D/{date_folder}'
                    os.makedirs(plot_dir, exist_ok=True)
                    safe_time = str(block_time).replace(':', '').replace('-', '').replace('T', '_')
                    
                    # Transform only the Alpha columns to log10
                    plot_samples = np.copy(samples)
                    plot_samples[:, 0:3] = np.log10(plot_samples[:, 0:3])
                    
                    # Update labels for 5D
                    labels = [r"$\log_{10}(\alpha_{true})$", r"$\log_{10}(\alpha_{eff,x})$", 
                              r"$\log_{10}(\alpha_{eff,y})$", r"$\gamma_{int,x}$", r"$\gamma_{int,y}$"]
                              
                    fig = corner.corner(plot_samples, labels=labels, 
                                        quantiles=[0.16, 0.5, 0.84], show_titles=True)
                    plt.suptitle(f"5D MCMC Posterior: {block_time}", y=1.05, fontsize=14)
                    plt.savefig(f'{plot_dir}/5D_corner_{safe_time}.png', dpi=150, bbox_inches='tight')
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
            
