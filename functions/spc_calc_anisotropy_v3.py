import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr
import emcee
import corner

from numba import njit
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde

from . import diagnostic_plots as plots

from scipy.ndimage import label

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

def bayesian_anisotropy_grid_solver(vz_fit, vtrans_fit, weights, bz, sigma_vz, uncert_map, flow_angle, 
                                    alpha_min=0.1, alpha_max=10.0, grid_size=400):
    
    alphas = np.geomspace(alpha_min, alpha_max, grid_size)
    prior = np.ones_like(alphas) 
    
    # 1. Forward Model 
    term1 = 1.0 - alphas
    term2 = 1.0 - term1 * (1.0 - bz**2)
    
    # Calculate the expected slopes (m) and V_trans line
    expected_slopes = (term1 * bz) / term2
    expected_vtrans = expected_slopes[:, None] * vz_fit[None, :]
    
    # Calculate the base data scatter (from the empirical fit)
    empirical_slope = np.sum(weights * vtrans_fit * vz_fit) / np.sum(weights * vz_fit**2)
    residuals = vtrans_fit - (empirical_slope * vz_fit)
    sigma_scatter_sq = np.maximum(np.average(residuals**2, weights=weights), 1e-6)
    
    # --- NEW: EFFECTIVE VARIANCE FROM Vz BINS ---
    # Propagate the error: Variance contribution = m(alpha)^2 * sigma_vz^2
    # expected_slopes is 1D (400,), sigma_vz is 1D (N data points)
    # The resulting var_from_vz is a 2D matrix (400 x N)
    var_from_vz = (expected_slopes[:, None]**2) * (sigma_vz[None, :])
    
    # Total Variance = Base scatter + Vz variance
    total_var = sigma_scatter_sq + var_from_vz
    # --------------------------------------------
    
    # Calculate Log-Likelihood INCLUSIVE of the changing variance penalty
    # Notice we MUST include the -0.5 * np.log(total_var) term now!
    log_likelihood = np.sum(
        weights * (-0.5 * ((vtrans_fit - expected_vtrans)**2 / total_var) - 0.5 * np.log(total_var)),
        axis=1
    )
    
    # Create the raw statistical posterior
    log_stat_post = log_likelihood 
    log_stat_post[np.isnan(log_stat_post)] = -np.inf
    
    valid_max = np.max(log_stat_post)
    if np.isinf(valid_max): valid_max = 0.0
        
    log_stat_post -= valid_max
    stat_posterior = np.exp(log_stat_post) * prior
    
    # Normalize the statistical posterior
    area_stat = cumulative_trapezoid(stat_posterior, alphas, initial=0)[-1]
    if area_stat > 0:
        stat_posterior /= area_stat
    else:
        return np.nan, np.nan, np.nan, stat_posterior, log_stat_post, alphas

    # ====================================================================
    # 2. CONVOLUTION (Smear the statistics with the Systematic map)
    # ====================================================================
    try:
        angle_list, ani_list, delta_plus_map, delta_minus_map, bias_map = uncert_map
    except ValueError:
        angle_list, ani_list, delta_plus_map, delta_minus_map = uncert_map
        bias_map = np.zeros_like(delta_plus_map)
        print("Warning: Old uncertainty map detected. Defaulting to zero systematic bias.")
        
    idx_angle = np.nanargmin(np.abs(np.abs(flow_angle) - angle_list))
    
    dp_grid = np.interp(alphas, ani_list, delta_plus_map[idx_angle, :])
    dm_grid = np.interp(alphas, ani_list, delta_minus_map[idx_angle, :])
    
    diff = alphas[:, None] - alphas[None, :]
    sigma_matrix = np.where(diff > 0, dp_grid[None, :], np.abs(dm_grid[None, :]))
    sigma_matrix = np.maximum(sigma_matrix, 1e-3) 
    
    kernel = np.exp(-0.5 * (diff / sigma_matrix)**2)
    
    # Normalize the kernel
    kernel_sums = np.sum(kernel, axis=0)
    kernel_sums[kernel_sums == 0] = 1.0  
    kernel = kernel / kernel_sums[None, :]
    
    convolved_posterior = kernel @ stat_posterior
    
    area_conv = cumulative_trapezoid(convolved_posterior, alphas, initial=0)[-1]
    if area_conv > 0:
        convolved_posterior /= area_conv
    
    log_convolved_posterior = np.log(np.maximum(convolved_posterior, 1e-300))
    
    # ====================================================================
    # 3. METRICS AND QC GATES
    # ====================================================================
    cdf = cumulative_trapezoid(convolved_posterior, alphas, initial=0)
    cdf /= cdf[-1] 
    
    alpha_16 = np.interp(0.16, cdf, alphas)
    alpha_50 = np.interp(0.50, cdf, alphas)
    alpha_84 = np.interp(0.84, cdf, alphas)
    
    if (alpha_50 < alpha_min * 2.0) or (alpha_50 > alpha_max * 0.98):
        return np.nan, np.nan, np.nan, convolved_posterior, log_convolved_posterior, alphas

    sig_minus = alpha_50 - alpha_16
    sig_plus = alpha_84 - alpha_50
    
    return alpha_50, sig_plus, sig_minus, convolved_posterior, log_convolved_posterior, alphas

def compute_bayesian_average(log_post_chi, log_post_psi, alphas):
    """
    Computes the combined average anisotropy by multiplying the Bayesian
    posteriors from the X and Y planes (which equals adding their log-posteriors).
    """
    # 1. Add the log-posteriors to get the joint log-posterior
    joint_log_post = log_post_chi + log_post_psi
    
    # 2. Exponentiate to get the raw probability curve
    # (Subtracting the max first prevents np.exp() from overflowing to infinity)
    joint_log_post -= np.nanmax(joint_log_post)
    joint_posterior = np.exp(joint_log_post)
    
    # 3. Normalize the area to exactly 1.0 (so it's a true PDF)
    joint_posterior /= np.trapz(joint_posterior, alphas)
    
    # 4. Extract the physical metrics from the new Joint Distribution
    # cdf = np.cumtrapz(joint_posterior, alphas, initial=0)
    cdf = cumulative_trapezoid(joint_posterior, alphas, initial=0)
    cdf /= cdf[-1]
    
    alpha_16 = np.interp(0.16, cdf, alphas)
    alpha_50 = np.interp(0.50, cdf, alphas) # This is your new "Average"
    alpha_84 = np.interp(0.84, cdf, alphas)
    
    sig_minus = alpha_50 - alpha_16
    sig_plus = alpha_84 - alpha_50
    
    return alpha_50, sig_plus, sig_minus, joint_posterior

def compute_bayesian_results(alpha_chi, sig_plus_chi, sig_minus_chi,
                             alpha_psi, sig_plus_psi, sig_minus_psi,
                             avg_alpha, avg_sig_plus, avg_sig_minus,
                             Nx_calc, Ny_calc, Nz_calc, Bvec, 
                             peak_azimuth, peak_elevation, timestamp, ds,
                             chi_sq_joint=np.nan, chi_sq_x=np.nan, chi_sq_y=np.nan,
                             norm_res_sq_x=None, norm_res_sq_y=None): 
    
    # Handle the NaN arrays for skipped timestamps
    if norm_res_sq_x is None:
        norm_res_sq_x = np.full(get_data(ds, "vel_mid").shape[1], np.nan)
    if norm_res_sq_y is None:
        norm_res_sq_y = np.full(get_data(ds, "vel_mid").shape[1], np.nan)

    # Package everything in one dict
    data_vars = dict(
        T_ani_avg             = avg_alpha,
        T_ani_avg_delta_plus  = avg_sig_plus,
        T_ani_avg_delta_minus = avg_sig_minus,

        T_ani_chi             = alpha_chi,
        T_ani_chi_delta_plus  = sig_plus_chi,
        T_ani_chi_delta_minus = sig_minus_chi,

        T_ani_psi             = alpha_psi,
        T_ani_psi_delta_plus  = sig_plus_psi,
        T_ani_psi_delta_minus = sig_minus_psi,
        
        chi_sq_joint          = chi_sq_joint,
        chi_sq_x              = chi_sq_x,
        chi_sq_y              = chi_sq_y,
        
        # Save the full arrays of individual squared residuals
        norm_res_sq_x         = (["dim"], norm_res_sq_x),
        norm_res_sq_y         = (["dim"], norm_res_sq_y),

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
        attrs=dict(Note=f"Bayesian Fitting Results for {get_data(ds, 'epoch')[0]} to {get_data(ds, 'epoch')[-1]}")
    )


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

def get_contiguous_peak_mask(weights, threshold):
    """
    Takes a 2D array of VDF weights (time, velocity) and returns a boolean mask
    keeping only the contiguous points around the peak of each row.
    """
    # 1. Create the raw threshold mask
    raw_mask = np.isfinite(weights) & (weights > threshold)
    
    # 2. The Magic Matrix: 
    # This 3x3 array tells scipy to only group pixels horizontally (velocity bins)
    # and NEVER group pixels vertically (adjacent time steps).
    structure = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]])
    
    # Label the islands
    labeled_array, _ = label(raw_mask, structure=structure)
    
    # 3. Find the exact index of the peak for each row
    peak_indices = np.zeros(weights.shape[0], dtype=int)
    valid_rows = np.any(np.isfinite(weights), axis=1) # Protect against all-NaN rows
    peak_indices[valid_rows] = np.nanargmax(weights[valid_rows], axis=1)
    
    # 4. Extract the island label that belongs to the peak
    row_indices = np.arange(weights.shape[0])
    peak_labels = labeled_array[row_indices, peak_indices]
    
    # 5. Build the clean mask: match the peak's label and ignore 0 (background)
    clean_mask = raw_mask & (labeled_array == peak_labels[:, None]) & (labeled_array > 0)
    
    return clean_mask

@njit
def log_likelihood_hierarchical(theta, vzbx, vx, vzby, vy, weights, bz):
    alpha_true, alpha_eff_x, alpha_eff_y, log_sigma_x, log_sigma_y = theta
    
    var_x = np.exp(2.0 * log_sigma_x)
    var_y = np.exp(2.0 * log_sigma_y)
    
    expected_slope_x = ((1.0 - alpha_eff_x) * bz) / (1.0 - (1.0 - alpha_eff_x) * (1.0 - bz**2))
    expected_slope_y = ((1.0 - alpha_eff_y) * bz) / (1.0 - (1.0 - alpha_eff_y) * (1.0 - bz**2))
    
    # --- NEW: Dynamically calculate the bulk transverse velocity offset (y-intercept) ---
    sum_w = np.sum(weights)
    intercept_x = np.sum(weights * (vx - expected_slope_x * vzbx)) / sum_w
    intercept_y = np.sum(weights * (vy - expected_slope_y * vzby)) / sum_w
    
    # Add the intercept to the theoretical model
    mu_x = expected_slope_x * vzbx + intercept_x
    mu_y = expected_slope_y * vzby + intercept_y
    # ------------------------------------------------------------------------------------
    
    L_x = np.sum(weights * (((vx - mu_x)**2 / var_x) + 1.837877 + 2.0 * log_sigma_x))
    L_y = np.sum(weights * (((vy - mu_y)**2 / var_y) + 1.837877 + 2.0 * log_sigma_y))
    
    return -0.5 * (L_x + L_y)


@njit
def log_prior_hierarchical(theta, ani_list, dp_slice_x, dm_slice_x, dp_slice_y, dm_slice_y):
    alpha_true, alpha_eff_x, alpha_eff_y, log_sigma_x, log_sigma_y = theta
    
    # 1. Uniform Bounds for all variables
    if not (0.1 < alpha_true < 10.0): return -np.inf
    if not (0.1 < alpha_eff_x < 10.0): return -np.inf
    if not (0.1 < alpha_eff_y < 10.0): return -np.inf
    if not (-10.0 < log_sigma_x < 5.0) or not (-10.0 < log_sigma_y < 5.0): return -np.inf
        
    # 2. Look up the systematic uncertainties evaluated at the TRUE alpha
    dp_x = np.interp(alpha_true, ani_list, dp_slice_x)
    dm_x = np.interp(alpha_true, ani_list, dm_slice_x)
    dp_y = np.interp(alpha_true, ani_list, dp_slice_y)
    dm_y = np.interp(alpha_true, ani_list, dm_slice_y)
    
    # 3. Calculate Split-Normal (Asymmetric Gaussian) Prior for X
    sig_x = dp_x if alpha_eff_x > alpha_true else np.abs(dm_x)
    sig_x = max(sig_x, 1e-3)
    # We must include the -np.log(dp + dm) normalization term because it changes depending on alpha_true!
    log_P_eff_x = -np.log(dp_x + np.abs(dm_x)) - 0.5 * ((alpha_eff_x - alpha_true) / sig_x)**2
    
    # 4. Calculate Split-Normal Prior for Y
    sig_y = dp_y if alpha_eff_y > alpha_true else np.abs(dm_y)
    sig_y = max(sig_y, 1e-3)
    log_P_eff_y = -np.log(dp_y + np.abs(dm_y)) - 0.5 * ((alpha_eff_y - alpha_true) / sig_y)**2
    
    return log_P_eff_x + log_P_eff_y

@njit
def log_probability_hierarchical(theta, vzbx, vx, vzby, vy, weights, bz, ani_list, dp_slice_x, dm_slice_x, dp_slice_y, dm_slice_y):
    lp = log_prior_hierarchical(theta, ani_list, dp_slice_x, dm_slice_x, dp_slice_y, dm_slice_y)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_hierarchical(theta, vzbx, vx, vzby, vy, weights, bz)

def run_emcee_hierarchical(vzbx, vx, vzby, vy, weights, bz, uncert_map, xangle, yangle):
    """
    Executes the 5D emcee MCMC sampler for the Joint Hierarchical Model.
    """
    # 1. Quick empirical guesses to start the walkers in a reasonable location
    sum_w = np.sum(weights)
    cx = np.sum(weights * vzbx) / sum_w
    cy = np.sum(weights * vx) / sum_w
    
    # Centroid-shifted slope (unhinged from the origin)
    slope_x = np.sum(weights * (vzbx - cx) * (vx - cy)) / max(np.sum(weights * (vzbx - cx)**2), 1e-6)
    
    emp_alpha = (-slope_x / (slope_x * (1. - bz**2) + bz)) + 1
    emp_alpha = np.clip(emp_alpha, 0.15, 9.85)

    # 2. Pre-slice the lookup tables for BOTH flow angles
    angle_list, ani_list, dp_map, dm_map = uncert_map
    
    idx_x = np.nanargmin(np.abs(np.abs(xangle) - angle_list))
    dp_slice_x = dp_map[idx_x, :]
    dm_slice_x = dm_map[idx_x, :]
    
    idx_y = np.nanargmin(np.abs(np.abs(yangle) - angle_list))
    dp_slice_y = dp_map[idx_y, :]
    dm_slice_y = dm_map[idx_y, :]
    
    # 3. Setup 5D Walkers (Using 32 walkers for a 5D space for better coverage)
    nwalkers = 32
    ndim = 5
    
    pos_alpha_true = emp_alpha + 1e-2 * np.random.randn(nwalkers)
    pos_alpha_eff_x = emp_alpha + 1e-2 * np.random.randn(nwalkers)
    pos_alpha_eff_y = emp_alpha + 1e-2 * np.random.randn(nwalkers)
    pos_log_sig_x = -2.0 + 0.1 * np.random.randn(nwalkers)
    pos_log_sig_y = -2.0 + 0.1 * np.random.randn(nwalkers)
    
    pos = np.vstack((pos_alpha_true, pos_alpha_eff_x, pos_alpha_eff_y, pos_log_sig_x, pos_log_sig_y)).T
    
    # 4. Initialize and Run Sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability_hierarchical, 
        args = (vzbx, vx, vzby, vy, weights, bz, ani_list, dp_slice_x, dm_slice_x, dp_slice_y, dm_slice_y)
    )
    
    # Increased steps slightly to allow the 5D space to properly burn-in
    sampler.run_mcmc(pos, 150, progress=False)
    
    # We only care about column 0 (alpha_true) for the final physical answer!
    samples = sampler.get_chain(discard=50, flat=True)
    samples_true = samples[:,0]
    samples_eff_x = samples[:,1]
    samples_eff_y = samples[:,2]
    
    # 5. Extract Asymmetric Error Bounds
    alpha_16, alpha_50, alpha_84 = np.percentile(samples_true, [16, 50, 84])
    
    sig_minus = alpha_50 - alpha_16
    sig_plus = alpha_84 - alpha_50

    return alpha_50, sig_plus, sig_minus, samples

@njit
def log_likelihood_hierarchical_3d(theta, vzbx, vx, vzby, vy, bz, var_x_array, var_y_array):
    alpha_true, alpha_eff_x, alpha_eff_y = theta
    
    expected_slope_x = ((1.0 - alpha_eff_x) * bz) / (1.0 - (1.0 - alpha_eff_x) * (1.0 - bz**2))
    expected_slope_y = ((1.0 - alpha_eff_y) * bz) / (1.0 - (1.0 - alpha_eff_y) * (1.0 - bz**2))
    
    # Use strict Inverse Variance Weights (W = 1/sigma^2)
    W_x = 1.0 / var_x_array
    W_y = 1.0 / var_y_array
    
    intercept_x = np.sum(W_x * (vx - expected_slope_x * vzbx)) / np.sum(W_x)
    intercept_y = np.sum(W_y * (vy - expected_slope_y * vzby)) / np.sum(W_y)
    
    mu_x = expected_slope_x * vzbx + intercept_x
    mu_y = expected_slope_y * vzby + intercept_y
    
    # Fractional weights removed! Variance strictly dictates the penalty.
    L_x = np.sum(((vx - mu_x)**2 / var_x_array) + 1.837877 + np.log(var_x_array))
    L_y = np.sum(((vy - mu_y)**2 / var_y_array) + 1.837877 + np.log(var_y_array))
    
    return -0.5 * (L_x + L_y)

@njit
def log_prior_hierarchical_3d(theta, ani_list, dp_slice_x, dm_slice_x, dp_slice_y, dm_slice_y):
    alpha_true, alpha_eff_x, alpha_eff_y = theta
    
    if not (0.1 < alpha_true < 10.0): return -np.inf
    if not (0.1 < alpha_eff_x < 10.0): return -np.inf
    if not (0.1 < alpha_eff_y < 10.0): return -np.inf
        
    dp_x = np.interp(alpha_true, ani_list, dp_slice_x)
    dm_x = np.interp(alpha_true, ani_list, dm_slice_x)
    dp_y = np.interp(alpha_true, ani_list, dp_slice_y)
    dm_y = np.interp(alpha_true, ani_list, dm_slice_y)
    
    sig_x = dp_x if alpha_eff_x > alpha_true else np.abs(dm_x)
    sig_x = max(sig_x, 1e-3)
    log_P_eff_x = -np.log(dp_x + np.abs(dm_x)) - 0.5 * ((alpha_eff_x - alpha_true) / sig_x)**2
    
    sig_y = dp_y if alpha_eff_y > alpha_true else np.abs(dm_y)
    sig_y = max(sig_y, 1e-3)
    log_P_eff_y = -np.log(dp_y + np.abs(dm_y)) - 0.5 * ((alpha_eff_y - alpha_true) / sig_y)**2
    
    return log_P_eff_x + log_P_eff_y

@njit
def log_probability_hierarchical_3d(theta, vzbx, vx, vzby, vy, bz, ani_list, dp_slice_x, dm_slice_x, dp_slice_y, dm_slice_y, var_x_array, var_y_array):
    lp = log_prior_hierarchical_3d(theta, ani_list, dp_slice_x, dm_slice_x, dp_slice_y, dm_slice_y)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_hierarchical_3d(theta, vzbx, vx, vzby, vy, bz, var_x_array, var_y_array)

def run_emcee_hierarchical_3d(vzbx, vx, vzby, vy, bz, uncert_map, xangle, yangle, var_x_array, var_y_array):
    # Use Inverse Variance for the starting guess
    W_x = 1.0 / var_x_array
    sum_W_x = np.sum(W_x)
    cx = np.sum(W_x * vzbx) / sum_W_x
    cy = np.sum(W_x * vx) / sum_W_x
    
    slope_x = np.sum(W_x * (vzbx - cx) * (vx - cy)) / max(np.sum(W_x * (vzbx - cx)**2), 1e-6)
    
    emp_alpha = (-slope_x / (slope_x * (1. - bz**2) + bz)) + 1
    emp_alpha = np.clip(emp_alpha, 0.15, 9.85)

    angle_list, ani_list, dp_map, dm_map = uncert_map
    
    idx_x = np.nanargmin(np.abs(np.abs(xangle) - angle_list))
    dp_slice_x = dp_map[idx_x, :]
    dm_slice_x = dm_map[idx_x, :]
    
    idx_y = np.nanargmin(np.abs(np.abs(yangle) - angle_list))
    dp_slice_y = dp_map[idx_y, :]
    dm_slice_y = dm_map[idx_y, :]
    
    nwalkers = 32
    ndim = 3  # Now strictly 3D!
    
    pos_alpha_true = emp_alpha + 1e-2 * np.random.randn(nwalkers)
    pos_alpha_eff_x = emp_alpha + 1e-2 * np.random.randn(nwalkers)
    pos_alpha_eff_y = emp_alpha + 1e-2 * np.random.randn(nwalkers)
    
    pos = np.vstack((pos_alpha_true, pos_alpha_eff_x, pos_alpha_eff_y)).T
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability_hierarchical_3d, 
        args = (vzbx, vx, vzby, vy, bz, ani_list, dp_slice_x, dm_slice_x, dp_slice_y, dm_slice_y, var_x_array, var_y_array)
    )
    
    sampler.run_mcmc(pos, 250, progress=False)
    
    samples = sampler.get_chain(discard=100, flat=True)
    samples_true = samples[:,0]
    
    alpha_16, alpha_50, alpha_84 = np.percentile(samples_true, [16, 50, 84])
    sig_minus = alpha_50 - alpha_16
    sig_plus = alpha_84 - alpha_50

    return alpha_50, sig_plus, sig_minus, samples

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
    try:
        # Try unpacking the new 5-element map
        angle_list, ani_list, anisotropy_delta_plus, anisotropy_delta_minus, bias_map = uncertainty_map
    except ValueError:
        # Fall back to the old 4-element map
        angle_list, ani_list, anisotropy_delta_plus, anisotropy_delta_minus = uncertainty_map
        
        # Create a dummy bias map filled with zeros so downstream math doesn't break
        bias_map = np.zeros_like(anisotropy_delta_plus)
        print("Warning: Old uncertainty map detected. Defaulting to zero systematic bias.")

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
    bias        = bias_map[mapped_angle_index, mapped_anisotropy_index]

    # Now get the standard deviation
    part_tani_part_slope = -Nz / ((1 - Nz**2) * fit_vals[0] + Nz)**2
    part_tani_part_Nz_chi = (fit_vals[0] - 2. * Nz * fit_vals[0]**2)/ ((1 - Nz**2) * fit_vals[0] + Nz)**2 

    std = np.sqrt((part_tani_part_slope * slope_error)**2 + (part_tani_part_Nz_chi * field_error)**2)

    delta_mag = np.sqrt(delta_plus**2 + delta_minus**2 + std**2)

    components = [(part_tani_part_slope * slope_error)**2, (part_tani_part_Nz_chi * field_error)**2]

    return(delta_plus, delta_minus, bias, std, delta_mag, components)

def _get_uncertainty2(fit_vals, slope_error, Nz, field_error, flow_angle, calc_anisotropy, uncertainty_map):
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

    delta_mag = np.sqrt(np.max((np.abs(delta_plus), np.abs(delta_minus)))**2 + std**2)

    components = [(part_tani_part_slope * slope_error)**2, (part_tani_part_Nz_chi * field_error)**2]

    return(delta_plus, delta_minus, std, delta_mag, components)

def compute_anisotropy(params_chi, slope_err_chi, Tani_chi,
                       params_psi, slope_err_psi, Tani_psi,
                       Nz_calc, xangle, yangle, uncertainty_map, Bmag):

    # helper to unpack uncertainty for one direction
    def anisotropy_uncertainty(params, slope_err, angle, Tani, Nz, Bmag):
        sigma_b = ((1 - Nz**2)/Bmag**2)
        # sigma_b = 0
        delta_plus, delta_minus, std, delta_mag, components = _get_uncertainty(
            params, slope_err, Nz_calc, sigma_b, angle, Tani, uncertainty_map
        )
        return {
            "anisotropy": Tani,
            "uncertainty": delta_mag,
            "uncertainty_delta_plus": delta_plus,
            "uncertainty_delta_minus": delta_minus,
            "fit_uncertainty": std**2,
            'components': components,
        }
    
        # helper to unpack uncertainty for one direction
    def anisotropy_uncertainty2(params, slope_err, angle, Tani, Nz, Bmag):
        sigma_b = ((1 - Nz**2)/Bmag**2)
        # sigma_b = 0
        delta_plus, delta_minus, std, delta_mag, components = _get_uncertainty2(
            params, slope_err, Nz_calc, sigma_b, angle, Tani, uncertainty_map
        )
        return {
            "anisotropy": Tani,
            "uncertainty": delta_mag,
            "uncertainty_delta_plus": delta_plus,
            "uncertainty_delta_minus": delta_minus,
            "fit_uncertainty": std**2,
            'components': components,
        }

    # OLD_AVERAGING = False
    # MODE = 'geometric'
    # MODE = 'symmetric'
    # MODE = 'asymmetric'
    MODE = 'stat_fit'

    
    if MODE == 'geometric':
        print(f"Using the {MODE} fitting routine.")
        chi = anisotropy_uncertainty(params_chi, slope_err_chi, xangle, Tani_chi, Nz_calc, Bmag)
        psi = anisotropy_uncertainty(params_psi, slope_err_psi, yangle, Tani_psi, Nz_calc, Bmag)

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
    elif MODE == 'symmetric':
        print(f"Using the {MODE} fitting routine.")
        # Get the uncertainty values. 
        chi = anisotropy_uncertainty2(params_chi, slope_err_chi, xangle, Tani_chi, Nz_calc, Bmag)
        psi = anisotropy_uncertainty2(params_psi, slope_err_psi, yangle, Tani_psi, Nz_calc, Bmag)

        ani_chi, sigma_chi = chi["anisotropy"], chi["uncertainty"]
        ani_psi, sigma_psi = psi["anisotropy"], psi["uncertainty"]

        sigma_2_chi = sigma_chi**2
        sigma_2_psi = sigma_psi**2

        w_chi = 1.0/sigma_2_chi
        w_psi = 1.0/sigma_2_psi

        w_sum = w_chi + w_psi

        ani_avg = (w_chi * ani_chi + w_psi * ani_psi) / w_sum

        sigma_avg = 1.0 / np.sqrt(w_sum)

        delta_plus_avg  = np.sqrt((w_chi * chi["uncertainty_delta_plus"]**2 + w_psi * psi["uncertainty_delta_plus"]**2) / w_sum)
        delta_minus_avg = np.sqrt((w_chi * chi["uncertainty_delta_minus"]**2 + w_psi * psi["uncertainty_delta_minus"]**2) / w_sum)

        fit_uncert_avg = np.sqrt((w_chi * chi["fit_uncertainty"]**2 + w_psi * psi["fit_uncertainty"]**2) / w_sum)

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
                "anisotropy": ani_avg,
                "uncertainty": sigma_avg,
                "uncertainty_delta_plus": delta_plus_avg,
                "uncertainty_delta_minus": delta_minus_avg,
                "fit_uncertainty": fit_uncert_avg,
            }

        return chi, psi, avg
    elif MODE == 'asymmetric':
        # print(f"Using the {MODE} fitting routine.")
        chi = anisotropy_uncertainty(params_chi, slope_err_chi, xangle, Tani_chi, Nz_calc, Bmag)
        psi = anisotropy_uncertainty(params_psi, slope_err_psi, yangle, Tani_psi, Nz_calc, Bmag)

        a_chi = chi["anisotropy"]
        a_psi = psi["anisotropy"]

        # Handle NaN cases gracefully
        if np.isnan(a_chi) and np.isnan(a_psi):
            avg = {k: np.nan for k in [
                "anisotropy", "uncertainty", "uncertainty_delta_plus",
                "uncertainty_delta_minus", "fit_uncertainty"
            ]}
        elif np.isnan(a_chi):
            avg = psi.copy()
        elif np.isnan(a_psi):
            avg = chi.copy()
        else:
            v_chi, gamma_chi = compute_moments(chi['uncertainty_delta_plus'], chi['uncertainty_delta_minus'], np.sqrt(chi['fit_uncertainty']))
            v_psi, gamma_psi = compute_moments(psi['uncertainty_delta_plus'], psi['uncertainty_delta_minus'], np.sqrt(psi['fit_uncertainty']))

            w_chi = 1.0/v_chi
            w_psi = 1.0/v_psi
        
            w_sum = w_chi + w_psi

            c_chi = w_chi/w_sum
            c_psi = w_psi/w_sum

            avg_ani = c_chi * a_chi + c_psi * a_psi

            v_avg = 1.0/w_sum
            gamma_avg = (c_chi**3) * gamma_chi + (c_psi**3) * gamma_psi

            sig_plus, sig_minus = extract_asymmetric_bounds(v_avg, gamma_avg)

            fit_uncert_avg_variance = (w_chi**2 * chi['fit_uncertainty'] + w_psi**2 * psi['fit_uncertainty']) / w_sum**2

            avg = {
                "anisotropy": avg_ani,
                "uncertainty": max(sig_plus, sig_minus), # Conservative 1D uncertainty
                "uncertainty_delta_plus": sig_plus,
                "uncertainty_delta_minus": sig_minus,
                "fit_uncertainty": fit_uncert_avg_variance,
            }
        return chi, psi, avg
    elif MODE == 'stat_fit':
        # print('Using the new mode called stat_fit')
        chi = anisotropy_uncertainty(params_chi, slope_err_chi, xangle, Tani_chi, Nz_calc, Bmag)
        psi = anisotropy_uncertainty(params_psi, slope_err_psi, yangle, Tani_psi, Nz_calc, Bmag)

        a_chi = chi["anisotropy"]
        a_psi = psi["anisotropy"]

        # Handle NaN cases gracefully
        if np.isnan(a_chi) and np.isnan(a_psi):
            avg = {k: np.nan for k in [
                "anisotropy", "uncertainty", "uncertainty_delta_plus",
                "uncertainty_delta_minus", "fit_uncertainty"
            ]}
        elif np.isnan(a_chi):
            avg = psi.copy()
        elif np.isnan(a_psi):
            avg = chi.copy()
        else:
            # Use only the pure statistical weighting.
            var_chi = max(chi['fit_uncertainty'], 1e-12)
            var_psi = max(psi['fit_uncertainty'], 1e-12)

            w_chi = 1.0/var_chi
            w_psi = 1.0/var_psi
            w_sum = w_chi + w_psi

            c_chi = w_chi / w_sum
            c_psi = w_psi / w_sum

            # The true anisotropy will now only use the fitting results.
            avg_ani = c_chi * a_chi + c_psi * a_psi

            var_avg_fit = 1.0/w_sum

            angle_list, ani_list, anisotropy_delta_plus, anisotropy_delta_minus = uncertainty_map

            mapped_ani_idx = np.nanargmin(np.abs(avg_ani - ani_list))
            idx_x = np.nanargmin(np.abs(np.abs(xangle) - angle_list))
            idx_y = np.nanargmin(np.abs(np.abs(yangle) - angle_list))

            # Grab the systematic bounds for the specific flow angles at the NEW anisotropy level
            dp_chi = anisotropy_delta_plus[idx_x, mapped_ani_idx]
            dm_chi = anisotropy_delta_minus[idx_x, mapped_ani_idx]
            dp_psi = anisotropy_delta_plus[idx_y, mapped_ani_idx]
            dm_psi = anisotropy_delta_minus[idx_y, mapped_ani_idx]
            
            # Blend the systematic bounds using the same data-quality weights
            avg_dp = c_chi * dp_chi + c_psi * dp_psi
            avg_dm = c_chi * dm_chi + c_psi * dm_psi
            
            # 5. COMPUTE FINAL ASYMMETRIC MOMENTS
            # Combine the purely statistical variance with the true systematic bounds
            v_avg, gamma_avg = compute_moments(avg_dp, avg_dm, np.sqrt(var_avg_fit))
            sig_plus, sig_minus = extract_asymmetric_bounds(v_avg, gamma_avg)
            
            avg = {
                "anisotropy": avg_ani,
                "uncertainty": max(sig_plus, sig_minus), # Conservative 1D envelope
                "uncertainty_delta_plus": sig_plus,
                "uncertainty_delta_minus": sig_minus,
                "fit_uncertainty": var_avg_fit,
            }    
        return chi, psi, avg
    else:
        return(0)

def compute_moments(delta_plus, delta_minus, std_fit):
    dp, dm = np.abs(delta_plus), np.abs(delta_minus)

    diff = dp - dm
    sum_sq = dp**2 + dm**2

    # Dimidiated Gaussian Moments
    v_eff = 0.5*sum_sq - (1.0 / (2.0 * np.pi)) * diff**2
    gamma = (1.0 / np.sqrt(2.0 * np.pi)) * (2.0 * (dp**3 - dm**3) - 1.5 * diff * sum_sq + (1.0 / np.pi) * diff**3)

    v_tot = v_eff + std_fit**2
    
    return v_tot, gamma

def extract_asymmetric_bounds(v_avg, gamma_avg):
    if np.isclose(gamma_avg, 0.0):
        sigma = np.sqrt(v_avg)
        return sigma, sigma

    C = (5.0 / (2.0 * np.pi)) - 1.0
    p = (3.0 * v_avg) / C
    q = (-np.sqrt(2.0 * np.pi) * gamma_avg) / C
    
    discriminant = (q / 2.0)**2 + (p / 3.0)**3
    
    if discriminant > 0:
        # One real root
        u = -q / 2.0 + np.sqrt(discriminant)
        v = -q / 2.0 - np.sqrt(discriminant)
        D = np.cbrt(u) + np.cbrt(v)
    else:
        # Three real roots (Trigonometric fallback)
        val = (3.0 * q / (2.0 * p)) * np.sqrt(-3.0 / p)
        val = np.clip(val, -1.0, 1.0) # Prevent floating point errors in arccos
        theta = np.arccos(val)
        
        roots = [
            2.0 * np.sqrt(-p / 3.0) * np.cos((theta - 2.0 * np.pi * k) / 3.0)
            for k in range(3)
        ]
        valid_roots = [r for r in roots if np.sign(r) == np.sign(gamma_avg)]
        
        def cubic_residual(x):
            return abs(C * x**3 + 3.0 * v_avg * x - np.sqrt(2.0 * np.pi) * gamma_avg)
            
        # D = min(valid_roots, key=cubic_residual) if valid_roots else 0.0
        D = min(valid_roots, key=abs) if valid_roots else 0.0

    S = 2.0 * v_avg + (D**2) / np.pi
    
    # Extract bounds safely
    val_inside_sqrt = max(0.0, 2.0 * S - D**2) 
    sigma_plus = 0.5 * (np.sqrt(val_inside_sqrt) + D)
    sigma_minus = 0.5 * (np.sqrt(val_inside_sqrt) - D)
    
    return sigma_plus, sigma_minus    
  
def compute_anisotropy_results(params_chi, slope_err_chi, Tani_chi,
                               params_psi, slope_err_psi, Tani_psi,
                               Nz_calc, xangle, yangle, uncertainty_map,
                               Nx_calc, Ny_calc, r_sq_chi, r_sq_psi,
                               chi_sq_chi, chi_sq_psi, peak_azimuth,
                               peak_elevation, timestamp, ds, Bvec):

    chi, psi, avg = compute_anisotropy(
        params_chi, slope_err_chi, Tani_chi,
        params_psi, slope_err_psi, Tani_psi,
        Nz_calc, xangle, yangle, uncertainty_map, np.linalg.norm(Bvec)
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

        T_chi_comps0           = chi["components"][0],
        T_chi_comps1           = chi["components"][1],

        T_ani_psi             = psi["anisotropy"],
        T_ani_psi_uncert      = psi["uncertainty"],
        T_ani_psi_delta_plus  = psi["uncertainty_delta_plus"],
        T_ani_psi_delta_minus = psi["uncertainty_delta_minus"],

        T_psi_comps0           = psi["components"][0],
        T_psi_comps1           = psi["components"][1],

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

# def compute_bayesian_results(alpha_chi, sig_plus_chi, sig_minus_chi,
#                              alpha_psi, sig_plus_psi, sig_minus_psi,
#                              avg_alpha, avg_sig_plus, avg_sig_minus,
#                              Nx_calc, Ny_calc, Nz_calc, Bvec, 
#                              peak_azimuth, peak_elevation, timestamp, ds,
#                              chi_sq_joint=np.nan): 
    
#     # Package everything in one dict, mimicking your linear output structure
#     data_vars = dict(
#         T_ani_avg             = avg_alpha,
#         T_ani_avg_delta_plus  = avg_sig_plus,
#         T_ani_avg_delta_minus = avg_sig_minus,

#         T_ani_chi             = alpha_chi,
#         T_ani_chi_delta_plus  = sig_plus_chi,
#         T_ani_chi_delta_minus = sig_minus_chi,

#         T_ani_psi             = alpha_psi,
#         T_ani_psi_delta_plus  = sig_plus_psi,
#         T_ani_psi_delta_minus = sig_minus_psi,
        
#         chi_sq_joint            = chi_sq_joint,

#         Nx = Nx_calc,
#         Ny = Ny_calc,
#         Nz = Nz_calc,

#         Bx = Bvec[0],
#         By = Bvec[1],
#         Bz = Bvec[2],

#         B_x_flow_angle = np.degrees(np.arctan(Nx_calc / Nz_calc)),
#         B_y_flow_angle = np.degrees(np.arctan(Ny_calc / Nz_calc)),

#         peak_azimuth   = peak_azimuth,
#         peak_elevation = peak_elevation
#     )

#     return xr.Dataset(
#         data_vars=data_vars,
#         coords=dict(time=[timestamp]),
#         attrs=dict(Note=f"Bayesian Fitting Results for {get_data(ds, 'epoch')[0]} to {get_data(ds, 'epoch')[-1]}")
#     )

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
        "np_fit", "np1_fit", "na_fit", "n3_fit", "np_moment",
        "wp_fit", "wp1_fit", "wa_fit", "w3_fit", "wp_moment"
    ]
    support_vector = [
        "vp_fit_SC", "vp1_fit_SC", "va_fit_SC", "v3_fit_SC", "vp_moment_SC"
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
        "np_fit", "np1_fit", "na_fit", "n3_fit", "np_moment",
        "wp_fit", "wp1_fit", "wa_fit", "w3_fit", "wp_moment"
    ]
    support_vector = [
        "vp_fit_SC", "vp1_fit_SC", "va_fit_SC", "v3_fit_SC", "vp_moment_SC"
    ]

    support_results = []

    for (t0, t1) in groups: #tqdm(groups, desc="Averaging support data"):
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
    # mask_weights1 = np.isfinite(weights) 
    # mask_weights2 = weights > threshold

    # mask_weights = mask_weights1 & mask_weights2
    mask_weights = get_contiguous_peak_mask(weights, threshold)

    if NO_WEIGHTS:
        mask_weights = np.isfinite(weights)     # We still only want finite values.
        weights = np.ones_like(weights)

    tani_chi = np.zeros(len(time))
    tani_psi = np.zeros(len(time))
    all_results = []
    for i in range(len(time)): #tqdm(range(len(time))):
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

        # if time[i] >= np.datetime64('2023-12-27T03:08:51'):
        #     plots.rdf_fit_diagnostic_plot_individual(ds, vx[i], vy[i], vz_bx[i], vz_by[i], rdf[i], time[i], result_ds)
        #     input()
    
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
    
    METHOD = 'Bayesian_MCMC'
    print(f'METHOD = {METHOD}')

    time = get_data(ds, "epoch")

    # Data extraction
    peak_az = get_data(ds, "peak_azimuth")
    peak_el = get_data(ds, "peak_elevation")
    vx = get_data(ds, "vel_x")
    vy = get_data(ds, "vel_y")
    vz = -get_data(ds, "vel_mid")
    vz_hi = -get_data(ds, "vel_hi")
    vz_lo = -get_data(ds, "vel_lo")
    bvec = np.atleast_2d(get_data(ds, "b_vec"))
    nvec = bvec / np.linalg.norm(bvec, axis=1)[:, None]
    vz_bx = vz * nvec[:, 0][:, None]
    vz_by = vz * nvec[:, 1][:, None]
    rdf = get_data(ds, "rdf")

    # Extract the Raw Plate Currents
    ia_curr = get_data(ds, "a_current")
    ib_curr = get_data(ds, "b_current")
    ic_curr = get_data(ds, "c_current")
    id_curr = get_data(ds, "d_current")


    # check_flux = False
    # if "diff_charge_flux_density" in ds:
        # charge_flux = get_data(ds, "diff_charge_flux_density")
        # check_flux = True

    weights = rdf / np.nanmax(rdf, axis=1)[:, None]
    # mask_weights = np.isfinite(weights) & (weights > threshold)
    mask_weights = get_contiguous_peak_mask(weights, threshold)
    if NO_WEIGHTS:
        mask_weights = np.isfinite(weights)
        weights = np.ones_like(weights)

    # --- Package into Dataset for slicing ---
    # We change what's packaged based on the B-field averaging method
    
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
        ia_curr = (["time", "dim"], ia_curr),
        ib_curr = (["time", "dim"], ib_curr),
        ic_curr = (["time", "dim"], ic_curr),
        id_curr = (["time", "dim"], id_curr),
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

    base_dict['vz_hi'] = (["time", "dim"], vz_hi)
    base_dict['vz_lo'] = (["time", "dim"], vz_lo)

    vz_geo_center = (vz_hi + vz_lo)/2.0
    vz_bias_sq = (vz - vz_geo_center)**2

    base_dict['vz_var'] = (["time", "dim"], ((vz_hi - vz_lo)**2)/12 + vz_bias_sq)

    base = xr.Dataset(base_dict, coords=dict(time=time))

    all_results = []

    # Case 1: groups not provided → one per timestamp
    if groups is None:
        groups = [(t, t) for t in time.values]
    count = 0
    for (t0, t1) in tqdm(groups):
        count += 1
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
            # block = block.drop_isel(time=block_bvec)
            block = block.isel(time=~block_bvec)
        

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
            # weight_mask = np.isfinite(weights) & (weights > threshold)
            weight_mask = get_contiguous_peak_mask(weights, threshold)

            valid_idx = weight_mask
            weight_fits = weights[valid_idx]
        else:
            # Stack all points in block
            valid_idx = block.mask.values
            weight_fits = block.weights.values[valid_idx]

        # Get vx and vy for fitting
        vx_fit = block.vx.values[valid_idx]
        vy_fit = block.vy.values[valid_idx]

        vz_var = block.vz_var.values[valid_idx]

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

        xangle = np.degrees(np.arctan(block_nvec_avg[0] / block_nvec_avg[2]))
        yangle = np.degrees(np.arctan(block_nvec_avg[1] / block_nvec_avg[2]))

        # Fits
        if METHOD == "Linear":
            results_x = find_slopes(linear_model, vzbx_fit, vx_fit, 1.0 / weight_fits, "lm")
            params_chi, cov_chi, chi_linear_fit, r_sq_chi, chi_sq_chi, slope_err_chi = results_x

            results_y = find_slopes(linear_model, vzby_fit, vy_fit, 1.0 / weight_fits, "lm")
            params_psi, cov_psi, psi_linear_fit, r_sq_psi, chi_sq_psi, slope_err_psi = results_y

            alpha_chi, Tani_chi = _calc_anisotropy_from_fits(params_chi, block_nvec_avg[2])
            alpha_psi, Tani_psi = _calc_anisotropy_from_fits(params_psi, block_nvec_avg[2])


            # # --- NEW: average L3 values for this block ---
            # l3_avgs = {}
            # list_of_keys = ["np_fit", "np1_fit", "n3_fit", "na_fit", "np_moment", 
            #                 "wp_fit", "wp1_fit", "w3_fit", "wa_fit", "wp_moment", 
            #                 "vp_fit_SC", "vp1_fit_SC", "v3_fit_SC", "va_fit_SC", "vp_moment_SC"]
            
            # for key in list_of_keys:
            #     if key in block:
            #         l3_avgs[key] = float(block[key].mean(skipna=True).values)
            #     else:
            #         l3_avgs[key] = np.nan

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
            continue
        
        elif METHOD == "Bayesian":
            # --- NEW: Project variance into X and Y axes ---
            var_vzbx = vz_var * (block_nvec_avg[0]**2)
            var_vzby = vz_var * (block_nvec_avg[1]**2)
            alpha_chi, sig_plus_chi, sig_minus_chi, post_chi, log_post_chi, grid_chi = bayesian_anisotropy_grid_solver(vzbx_fit, vx_fit, weight_fits, block_nvec_avg[2], var_vzbx, uncertainty_map, xangle)
            alpha_psi, sig_plus_psi, sig_minus_psi, post_psi, log_post_psi,  grid_psi = bayesian_anisotropy_grid_solver(vzby_fit, vy_fit, weight_fits, block_nvec_avg[2], var_vzby, uncertainty_map, yangle)

            valid_chi = not np.isnan(alpha_chi)
            valid_psi = not np.isnan(alpha_psi)

            if valid_chi and valid_psi:
                # Both planes are good! Do the true Bayesian Joint Average
                # Now because we are being a true Bayesian.
                avg_alpha, avg_sig_plus, avg_sig_minus, joint_post = compute_bayesian_average(log_post_chi, log_post_psi, grid_chi)
                
            elif valid_chi and not valid_psi:
                # print('Y-plane failed')
                # Y-plane failed or hit the boundary. Trust the X-plane entirely!
                avg_alpha, avg_sig_plus, avg_sig_minus = alpha_chi, sig_plus_chi, sig_minus_chi
                joint_post = np.exp(log_post_chi) # Convert log back to normal probability for plotting
                
            elif valid_psi and not valid_chi:
                # print('X-plane failed')
                # X-plane failed or hit the boundary. Trust the Y-plane entirely!
                avg_alpha, avg_sig_plus, avg_sig_minus = alpha_psi, sig_plus_psi, sig_minus_psi
                joint_post = np.exp(log_post_psi)
                
            else:
                # print('Both Failed')
                # Both planes completely failed. Throw the whole timestamp out.
                avg_alpha, avg_sig_plus, avg_sig_minus = np.nan, np.nan, np.nan
                joint_post = np.full_like(grid_chi, np.nan)

            # --- DIAGNOSTIC PLOTTING ---
            PLOT_POSTERIORS = False  # Toggle this to False when running the full 60 days!
            
            # Optional: Only plot every 100th point to save time, or specific times
            if PLOT_POSTERIORS and (count % 100 == 0): 
                

                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot the individual planes to see how they agree/disagree
                ax.plot(grid_chi, log_post_chi, color='blue', alpha=0.4, label='X-Z Plane')
                ax.plot(grid_psi, log_post_psi, color='orange', alpha=0.4, label='Y-Z Plane')
                
                # Plot the final combined Joint Posterior
                ax.plot(grid_chi, np.log(joint_post), color='black', linewidth=2, label='Joint Posterior')
                
                # Draw lines for the median and the physical error bounds
                ax.axvline(avg_alpha, color='red', linestyle='--', label=f'Median ($T_\perp/T_\parallel$): {avg_alpha:.2f}')
                ax.axvspan(avg_alpha - avg_sig_minus, avg_alpha + avg_sig_plus, 
                           color='red', alpha=0.2, label='Final Uncertainty Bounds')
                
                # Formatting
                ax.set_xscale('log')  # Because our grid is logarithmically spaced
                ax.set_xlim(0.1, 10.0)
                ax.set_xlabel('Temperature Anisotropy ($T_\perp / T_\parallel$)')
                ax.set_ylabel('Probability Density')
                ax.set_title(f'Log Bayesian Posterior | Valid Chi {valid_chi} | Valid Psi {valid_psi} \n {block_time}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Create a safe filename from the timestamp and save
                safe_time = str(block_time).replace(':', '').replace('-', '').replace('T', '_')
                plt.savefig(f'posterior_plots/posterior_{safe_time}.png', dpi=150, bbox_inches='tight')
                
                # CRITICAL: Close the figure to prevent your RAM from overloading
                plt.close(fig)

            result_ds = compute_bayesian_results(
                alpha_chi, sig_plus_chi, sig_minus_chi,
                alpha_psi, sig_plus_psi, sig_minus_psi,
                avg_alpha, avg_sig_plus, avg_sig_minus,
                block_nvec_avg[0],       # Nx_calc
                block_nvec_avg[1],       # Ny_calc
                block_nvec_avg[2],       # Nz_calc
                block_bvec_to_save,      # Bvec
                block.peak_az.mean().item(),
                block.peak_el.mean().item(),
                block_time,
                ds
            )
            
            all_results.append(result_ds)
            continue

        elif METHOD == "Bayesian_MCMC":
            sig_I = 1.0
            lamda = 1.009
            C     = lamda/2.

            # Extract the raw currents for valid points
            ia_fit = block.ia_curr.values[valid_idx]
            ib_fit = block.ib_curr.values[valid_idx]
            ic_fit = block.ic_curr.values[valid_idx]
            id_fit = block.id_curr.values[valid_idx]

            i_tot  = np.maximum(ia_fit + ib_fit + ic_fit + id_fit, 1e-6) # Ensure we do not get zeros!
            i_diff_x = (ia_fit + id_fit) - (ib_fit + ic_fit)
            i_diff_y = (ic_fit + id_fit) - (ia_fit + ib_fit)

            vz_fit = block.vz.values[valid_idx]
            vz_fit_abs = np.abs(block.vz.values[valid_idx])
            var_vz_fit = block.vz_var.values[valid_idx]

            # # Get TRUE Plasma Flow Angles using measured velocities
            theta_x_rad = np.arctan(vx_fit / vz_fit_abs)
            theta_y_rad = np.arctan(vy_fit / vz_fit_abs)

            # # Term 1: Vz Variance Projection
            # term1_x = var_vz_fit * (np.tan(theta_x_rad)**2)
            # term1_y = var_vz_fit * (np.tan(theta_y_rad)**2)
            cov_x = np.cov(vzbx_fit, vx_fit)
            emp_slope_x = cov_x[0, 1] / max(cov_x[0, 0], 1e-6)

            cov_y = np.cov(vzby_fit, vy_fit)
            emp_slope_y = cov_y[0, 1] / max(cov_y[0, 0], 1e-6)

            nx = block_nvec_avg[0]
            ny = block_nvec_avg[1]

            # Exact Partial Derivative: d(\Delta x)/dv_z = (v_x / v_z) - M_x * n_x
            deriv_x = (vx_fit / vz_fit) - (emp_slope_x * nx)
            deriv_y = (vy_fit / vz_fit) - (emp_slope_y * ny)

            # Term 1: Vz Variance Projection (Now with rigorous magnetic coupling!)
            term1_x = var_vz_fit * (deriv_x**2)
            term1_y = var_vz_fit * (deriv_y**2)
            
            geom_mod_x = 1.0 / (np.cos(theta_x_rad)**4)
            geom_mod_y = 1.0 / (np.cos(theta_y_rad)**4)

            prefix = (4.0 * (sig_I**2) * (vz_fit_abs**2) * (C**2)) / (i_tot**2)
            
            term2_x = prefix * geom_mod_x * (1.0 + (i_diff_x / i_tot)**2)
            term2_y = prefix * geom_mod_y * (1.0 + (i_diff_y / i_tot)**2)

            # Final Exact Hardware Variance
            var_x_array = np.maximum(term1_x + term2_x, 1e-4)
            var_y_array = np.maximum(term1_y + term2_y, 1e-4)

            avg_alpha, avg_sig_plus, avg_sig_minus, samples = run_emcee_hierarchical_3d(
                vzbx_fit, vx_fit, vzby_fit, vy_fit, 
                block_nvec_avg[2], uncertainty_map, xangle, yangle,
                var_x_array, var_y_array
            )

            # --- CALCULATE JOINT REDUCED CHI-SQUARED ---
            alpha_eff_x = np.median(samples[:, 1])
            alpha_eff_y = np.median(samples[:, 2])
            bz = block_nvec_avg[2]
            
            m_eff_x = ((1.0 - alpha_eff_x) * bz) / (1.0 - (1.0 - alpha_eff_x) * (1.0 - bz**2))
            m_eff_y = ((1.0 - alpha_eff_y) * bz) / (1.0 - (1.0 - alpha_eff_y) * (1.0 - bz**2))
            
            # Use Inverse Variance Weighting for the theoretical intercepts
            W_x = 1.0 / var_x_array
            W_y = 1.0 / var_y_array
            b_eff_x = np.sum(W_x * (vx_fit - m_eff_x * vzbx_fit)) / np.sum(W_x)
            b_eff_y = np.sum(W_y * (vy_fit - m_eff_y * vzby_fit)) / np.sum(W_y)
            
            # 1. Calculate the individual squared residuals for the valid points
            sq_res_x_valid = ((vx_fit - (m_eff_x * vzbx_fit + b_eff_x))**2) / var_x_array
            sq_res_y_valid = ((vy_fit - (m_eff_y * vzby_fit + b_eff_y))**2) / var_y_array
            
            # 2. Create the static length 1D NaN arrays 
            # (Using the second dimension of the block to match xarray's expected "dim" size)
            static_length = block.vx.shape[1]
            norm_res_sq_x = np.full(static_length, np.nan)
            norm_res_sq_y = np.full(static_length, np.nan)
            
            # 3. Fill the arrays sequentially from the front
            n_valid = len(sq_res_x_valid)
            if n_valid > 0:
                # Truncate in the extremely rare case n_valid > static_length
                fill_len = min(n_valid, static_length) 
                norm_res_sq_x[:fill_len] = sq_res_x_valid[:fill_len]
                norm_res_sq_y[:fill_len] = sq_res_y_valid[:fill_len]
            
            # 4. Calculate Scalar Chi-Squared values
            dof_single = max(len(vzbx_fit) - 2.0, 1.0)
            dof_joint = max((2.0 * len(vzbx_fit)) - 3.0, 1.0)
            
            chi_sq_x = np.sum(sq_res_x_valid) / dof_single
            chi_sq_y = np.sum(sq_res_y_valid) / dof_single
            chi_sq_joint = (np.sum(sq_res_x_valid) + np.sum(sq_res_y_valid)) / dof_joint
            # ------------------------------------------------------

            # Package the results
            result_ds = compute_bayesian_results(
                np.nan, np.nan, np.nan,  
                np.nan, np.nan, np.nan,  
                avg_alpha, avg_sig_plus, avg_sig_minus,
                block_nvec_avg[0],       
                block_nvec_avg[1],       
                block_nvec_avg[2],       
                block_bvec_to_save,      
                block.peak_az.mean().item(),
                block.peak_el.mean().item(),
                block_time,
                ds,
                chi_sq_joint=chi_sq_joint,
                chi_sq_x=chi_sq_x,
                chi_sq_y=chi_sq_y,
                norm_res_sq_x=norm_res_sq_x,
                norm_res_sq_y=norm_res_sq_y
            )

            # if (count % 100 == 0): 
                # print("This is the value I am looking at", var_x_array, var_y_array)
            PLOT_POSTERIORS = False

            if PLOT_POSTERIORS and (count % 50 == 0): 
                import os
                date_folder = str(block_time)[:10] 
                plot_dir = f'posterior_plots/{date_folder}'
                plot_dir_2 = f'diagnostic_plots/{date_folder}'
                os.makedirs(plot_dir, exist_ok=True)

                safe_time = str(block_time).replace(':', '').replace('-', '').replace('T', '_')
                plots.rdf_fit_diagnostic_plot_mcmc(ds, block, block_time, result_ds, samples, var_x_array, var_y_array, save_dir=plot_dir_2, safe_time=safe_time)
                
                plot_samples = samples.copy()
                plot_samples[:, 0] = np.log10(plot_samples[:, 0]) 
                plot_samples[:, 1] = np.log10(plot_samples[:, 1]) 
                plot_samples[:, 2] = np.log10(plot_samples[:, 2]) 
                
                labels = [
                    r"$\log_{10}(\alpha_{true})$", 
                    r"$\log_{10}(\alpha_{eff,x})$", 
                    r"$\log_{10}(\alpha_{eff,y})$"
                ]
                
                log_truth = np.log10(avg_alpha)
                
                fig_corner = corner.corner(
                    plot_samples, 
                    labels=labels, 
                    quantiles=[0.16, 0.5, 0.84], 
                    show_titles=True, 
                    title_kwargs={"fontsize": 10},
                    truths=[log_truth, None, None], 
                    truth_color='red'
                )
                
                fig_corner.suptitle(f"3D Parameter Space (Log Scaled) | {block_time}", fontsize=14, y=1.05)
                plt.savefig(f'{plot_dir}/corner_{safe_time}.png', dpi=150, bbox_inches='tight')
                plt.close(fig_corner)
            
            all_results.append(result_ds)
            continue
            # print("5D!")

            # # The 5D Hierarchical sampler dynamically balances the X and Y planes!
            # avg_alpha, avg_sig_plus, avg_sig_minus, samples = run_emcee_hierarchical(
            #     vzbx_fit, vx_fit, vzby_fit, vy_fit, 
            #     weight_fits, block_nvec_avg[2], uncertainty_map, xangle, yangle
            # )

            # # --- NEW: CALCULATE JOINT REDUCED CHI-SQUARED ---
            # # 1. Extract Median Parameters from Samples
            # alpha_eff_x = np.median(samples[:, 1])
            # alpha_eff_y = np.median(samples[:, 2])
            # var_x = np.exp(2.0 * np.median(samples[:, 3]))
            # var_y = np.exp(2.0 * np.median(samples[:, 4]))
            # bz = block_nvec_avg[2]
            
            # # 2. Calculate Theoretical Slopes
            # m_eff_x = ((1.0 - alpha_eff_x) * bz) / (1.0 - (1.0 - alpha_eff_x) * (1.0 - bz**2))
            # m_eff_y = ((1.0 - alpha_eff_y) * bz) / (1.0 - (1.0 - alpha_eff_y) * (1.0 - bz**2))
            
            # # 3. Calculate Optimal Intercepts 
            # sum_w = np.sum(weight_fits)
            # b_eff_x = np.sum(weight_fits * (vx_fit - m_eff_x * vzbx_fit)) / sum_w
            # b_eff_y = np.sum(weight_fits * (vy_fit - m_eff_y * vzby_fit)) / sum_w
            
            # # 4. Calculate SINGLE Joint Reduced Chi-Squared
            # N_eff = 2.0 * sum_w
            # dof = max(N_eff - 5.0, 1.0)
            
            # sum_sq_x = np.sum(weight_fits * (vx_fit - (m_eff_x * vzbx_fit + b_eff_x))**2 / var_x)
            # sum_sq_y = np.sum(weight_fits * (vy_fit - (m_eff_y * vzby_fit + b_eff_y))**2 / var_y)
            
            # chi_sq_joint = (sum_sq_x + sum_sq_y) / dof
            # # ------------------------------------------

            # # Package the results
            # result_ds = compute_bayesian_results(
            #     np.nan, np.nan, np.nan,  # Skipped individual chi
            #     np.nan, np.nan, np.nan,  # Skipped individual psi
            #     avg_alpha, avg_sig_plus, avg_sig_minus,
            #     block_nvec_avg[0],       
            #     block_nvec_avg[1],       
            #     block_nvec_avg[2],       
            #     block_bvec_to_save,      
            #     block.peak_az.mean().item(),
            #     block.peak_el.mean().item(),
            #     block_time,
            #     ds,
            #     chi_sq_joint=chi_sq_joint,
            # )

            # PLOT_POSTERIORS = False

            # if PLOT_POSTERIORS and (count % 100 == 0): 
            #     date_folder = str(block_time)[:10]  # Extracts just the 'YYYY-MM-DD'
            #     plot_dir = f'posterior_plots/{date_folder}'

            #     plots.rdf_fit_diagnostic_plot_mcmc_5D(ds, block, block_time, result_ds, samples)

            #     safe_time = str(block_time).replace(':', '').replace('-', '').replace('T', '_')
                
            #     # 1. MAKE A COPY of the samples to transform for the plot
            #     plot_samples = samples.copy()
                
            #     # 2. Convert the three alpha columns into log10 space
            #     # (Columns 3 and 4 are already ln(sigma), so we leave them alone)
            #     plot_samples[:, 0] = np.log10(plot_samples[:, 0]) # alpha_true
            #     plot_samples[:, 1] = np.log10(plot_samples[:, 1]) # alpha_eff_x
            #     plot_samples[:, 2] = np.log10(plot_samples[:, 2]) # alpha_eff_y
                
            #     # 3. Update the labels so the reader knows it is log10
            #     labels = [
            #         r"$\log_{10}(\alpha_{true})$", 
            #         r"$\log_{10}(\alpha_{eff,x})$", 
            #         r"$\log_{10}(\alpha_{eff,y})$", 
            #         r"$\ln(\sigma_x)$", 
            #         r"$\ln(\sigma_y)$"
            #     ]
                
            #     # 4. Make sure the 'Truth' red line is also in log10 space!
            #     log_truth = np.log10(avg_alpha)
                
            #     # Generate the corner plot with the transformed data
            #     fig_corner = corner.corner(
            #         plot_samples, 
            #         labels=labels, 
            #         quantiles=[0.16, 0.5, 0.84], 
            #         show_titles=True, 
            #         title_kwargs={"fontsize": 10},
            #         truths=[log_truth, None, None, None, None], 
            #         truth_color='red'
            #     )
                
            #     fig_corner.suptitle(f"5D Parameter Space (Log Scaled) | {block_time}", fontsize=14)
            #     plt.savefig(f'{plot_dir}/corner_{safe_time}.png', dpi=150, bbox_inches='tight')
            #     plt.close(fig_corner)
            # # ---------------------------
            # # ---------------------------


            
            # all_results.append(result_ds)
            continue            
        else:
            print("Failed to converge to a solution.")
            return(0)

    return xr.concat(all_results, dim="time")
