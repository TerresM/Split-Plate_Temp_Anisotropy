import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d

from matplotlib.colors import LogNorm

import cdflib.xarray

# ==========================================
# 1. DATA PRE-PROCESSING (Binning)
# ==========================================
def bin_max_uncertainty(angles_in, anis_in, delta_in, bins_theta=50, bins_ani=50):
    """
    Bins the input scattered data into a 2D grid.
    Returns the grid centers (Theta, Ani) and the MAX value in each bin.
    """
    # Define Bin Edges
    # Theta: Linear from 0 to 90
    theta_edges = np.linspace(0, 90, bins_theta + 1)
    
    # Anisotropy: Logarithmic from min to max
    ani_min = np.min(anis_in[anis_in > 0])
    ani_max = np.max(anis_in)
    ani_edges = np.logspace(np.log10(ani_min), np.log10(ani_max), bins_ani + 1)
    
    # Flatten inputs if they aren't already
    flat_theta = angles_in.ravel()
    flat_ani = anis_in.ravel()
    flat_delta = delta_in.ravel()
    
    # Perform 2D Binning with 'max' statistic
    # This finds the single highest uncertainty value in every grid cell
    H, x_edge, y_edge, _ = binned_statistic_2d(
        flat_theta, flat_ani, flat_delta, 
        statistic='max', 
        bins=[theta_edges, ani_edges]
    )
    
    # Calculate Bin Centers for the Fit
    theta_centers = (x_edge[:-1] + x_edge[1:]) / 2
    ani_centers_log = (np.log10(y_edge[:-1]) + np.log10(y_edge[1:])) / 2
    ani_centers = 10**ani_centers_log
    
    # Meshgrid of centers
    THETA_GRID, ANI_GRID = np.meshgrid(theta_centers, ani_centers, indexing='ij')
    
    # H is returned as shape (nx, ny), we need to match meshgrid
    # H[i, j] corresponds to theta[i] and ani[j]
    DELTA_GRID = H
    
    # Handle empty bins (NaNs) - simple interpolation or masking
    # For fitting, we can just mask them out.
    
    return THETA_GRID, ANI_GRID, DELTA_GRID

# ==========================================
# 1. THE ANALYTICAL MODEL
# ==========================================
def analytical_uncertainty_model(X, sigma_base, T_limit, exponent):
    """
    Parametric model for SPC Uncertainty.
    """
    theta_deg, A_true = X
    theta_rad = np.radians(theta_deg)
    
    S = np.sin(theta_rad)
    C = np.cos(theta_rad)
    
    # 1. Sensitivity (Geometric Term)
    denom_z = (C**2) + A_true * (S**2)
    dm_dA = -1.0 * (S * C) / (denom_z**2)
    sensitivity = np.abs(dm_dA)
    
    # 2. Resolution Penalty (Low Anisotropy)
    resolution_penalty = 1.0 / np.sqrt(denom_z)
    
    # 3. Aperture Clipping Penalty (High Anisotropy)
    T_x = (S**2) + A_true * (C**2)
    clipping_penalty = 1.0 + (T_x / T_limit)**exponent
    
    # 4. Total Noise Sigma
    sigma_m = sigma_base * resolution_penalty * clipping_penalty
    
    # 5. Predicted Uncertainty (Sigma_A)
    sigma_A = sigma_m / (sensitivity + 1e-9)
    
    # return (sigma_A / A_true) * 100 
    return sigma_A

def log_analytical_model(X, sigma_base, T_limit, exponent):
    """
    Wrapper that returns log10(Sigma_A) for the optimizer.
    """
    val = analytical_uncertainty_model(X, sigma_base, T_limit, exponent, )
    # Clip small values to avoid log(0)
    val = np.maximum(val, 1e-9) 
    return np.log10(val)

# ==========================================
# 2. FITTING FUNCTION
# ==========================================
def fit_surface(angles, anisotropies, delta_map, label="Uncertainty"):
    """
    Fits the analytical model to a specific uncertainty map (Plus or Minus).
    """
    print(f"Fitting {label} Model...")
    
    # Flatten Data for curve_fit
    THETA, ANI = np.meshgrid(angles, anisotropies)
    
    x_data = (THETA.flatten(), ANI.flatten())
    y_data = delta_map.flatten()
    
    # Filter valid data (remove NaNs and zeroes)
    mask = np.isfinite(y_data) & (y_data > 0)
    
    # Fit
    # Guesses: sigma=0.025, T_limit=15, exponent=4
    p0 = [0.025, 15.0, 4.0, 0.0]
    bounds = ([0.001, 1.0, 0.1, -3], [0.1, 100.0, 10.0, 2])
    
    popt, pcov = curve_fit(log_analytical_model, 
                           (x_data[0][mask], x_data[1][mask]), 
                           y_data[mask], 
                           p0=p0, bounds=bounds)
    
    print(f"  > Best Fit {label}:")
    print(f"    Sigma Base: {popt[0]:.5f}")
    print(f"    T_limit:    {popt[1]:.4f}")
    print(f"    Exponent:   {popt[2]:.4f}")
    
    return popt

def fit_surface_flat(angles, anisotropies, delta_map, label="Uncertainty"):
    """
    Fits the analytical model to a specific uncertainty map (Plus or Minus).
    """
    print(f"Fitting {label} Model...")
    
    x_data = (angles, anisotropies)
    y_data = delta_map
    # y_data = (delta_map / anisotropies) * 100
    
    # Filter valid data (remove NaNs and zeroes)
    mask = np.isfinite(y_data) & (y_data > 0)
    
    # Fit
    # Guesses: sigma=0.025, T_limit=15, exponent=4
    p0 = [0.025, 15.0, 4.0]
    bounds = ([0.001, 1.0, 0.1], [0.1, 100.0, 10.0])
    
    popt, pcov = curve_fit(log_analytical_model, #analytical_uncertainty_model, 
                           (x_data[0][mask], x_data[1][mask]), 
                           np.log10(y_data[mask]), 
                           p0=p0, bounds=bounds)
    
    print(f"  > Best Fit {label}:")
    print(f"    Sigma Base: {popt[0]:.5f}")
    print(f"    T_limit:    {popt[1]:.4f}")
    print(f"    Exponent:   {popt[2]:.4f}")
    
    return popt

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # LOAD DATA (Replace with your actual file loading)
    # ---------------------------------------------------------
    data = cdflib.xarray.cdf_to_xarray('Anisotropy_DELTA_PLUS-MINUS_newest.cdf')
    data = data.where((data != np.inf) & (data != -1e31))

    angles = data['Variable_4'].values
    anisotropies = data['Variable_5'].values
    d_plus = data['UncertaintyMap DELTA_PLUS'].values
    d_minus = data['UncertaintyMap DELTA_MINUS'].values

    # Only need angles less than 90
    angle_mask = angles < 90

    d_plus = d_plus[angle_mask, :]
    d_minus = d_minus[angle_mask, :]

    angles = angles[angle_mask]

    TH, AN = np.meshgrid(angles, anisotropies, indexing='ij')

    mask_p = np.isfinite(d_plus)
    mask_m = np.isfinite(d_minus)

    angle_plus, ani_plus, d_plus_bin = bin_max_uncertainty(TH[mask_p], AN[mask_p], d_plus[mask_p], bins_theta=50, bins_ani=50)
    angle_minus, ani_minus, d_minus_bin = bin_max_uncertainty(TH[mask_m], AN[mask_m], d_minus[mask_m], bins_theta=50, bins_ani=50)

    # -------------------------------------

    # 1. FIT UPPER BOUND (Delta Plus)
    # params_plus = fit_surface(angles, anisotropies, d_plus, label="Delta Plus (Upper)")
    params_plus = fit_surface_flat(angle_plus, ani_plus, d_plus_bin, label="Delta Plus (Upper)")
    
    # 2. FIT LOWER BOUND (Delta Minus)
    # params_minus = fit_surface(angles, anisotropies, d_minus, label="Delta Minus (Lower)")
    params_minus = fit_surface_flat(angle_minus, ani_minus, d_minus_bin, label="Delta Minus (Lower)")
    
    # Generate Model Maps
    map_plus_fit = analytical_uncertainty_model((TH, AN), *params_plus)
    map_minus_fit = analytical_uncertainty_model((TH, AN), *params_minus)

    # map_plus_fit = analytical_uncertainty_model((angle_plus, ani_plus), *params_plus)
    # map_minus_fit = analytical_uncertainty_model((angle_minus, ani_minus), *params_minus)

    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
    # # Row 1: Delta Plus
    # ax[0,0].pcolormesh(TH, AN, d_plus/AN*100, cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    # ax[0,0].set_title('Simulation: Delta Plus')
    # ax[0,0].set_yscale('log')
    # ax[0,0].scatter(TH, AN, c=d_plus, cmap='jet', norm=LogNorm(vmin=1, vmax=100))
    a0 = ax[0,0].pcolormesh(angle_plus, ani_plus, d_plus_bin/ani_plus*100, cmap='jet', norm=LogNorm(vmin=1, vmax=100))
    ax[0,0].set_title('Simulation: Delta Plus')
    ax[0,0].set_yscale('log')


    a1 = ax[0,1].pcolormesh(TH, AN, map_plus_fit/AN*100, cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    # a1 = ax[0,1].pcolormesh(TH, AN, map_plus_fit, cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    # ax[0,1].pcolormesh(angle_plus, ani_plus, map_plus_fit/ani_plus*100, cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    ax[0,1].set_title(f'Analytical Fit: Delta Plus\n($\sigma={params_plus[0]:.3f}, T={params_plus[1]:.1f}, n={params_plus[2]:.1f}$)')
    ax[0,1].set_yscale('log')
    
    # Row 2: Delta Minus
    # ax[1,0].pcolormesh(TH, AN, d_minus/AN*100, cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    # ax[1,0].scatter(TH, AN, c=d_minus/AN*100, cmap='jet', norm=LogNorm(vmin=1, vmax=100))
    a2 = ax[1,0].pcolormesh(angle_minus, ani_minus, d_minus_bin/ani_minus*100, cmap='jet', norm=LogNorm(vmin=1, vmax=100))
    ax[1,0].set_title('Simulation: Delta Minus')
    ax[1,0].set_yscale('log')
    
    a3 = ax[1,1].pcolormesh(TH, AN, map_minus_fit/AN*100, cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    # a3 = ax[1,1].pcolormesh(TH, AN, map_minus_fit, cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    # ax[1,1].pcolormesh(angle_minus, ani_minus, map_minus_fit/ani_minus*100, cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    ax[1,1].set_title(f'Analytical Fit: Delta Minus\n($\sigma={params_minus[0]:.3f}, T={params_minus[1]:.1f}, n={params_minus[2]:.1f}$)')
    ax[1,1].set_yscale('log')

    [ax[i,j].set_ylim(1e-1,10) for i in range(2) for j in range(2)]

    plt.colorbar(a0)
    plt.colorbar(a1)
    plt.colorbar(a2)
    plt.colorbar(a3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. PRINT FINAL LOOKUP FUNCTIONS
    print("\n" + "="*50)
    print("FINAL PARAMETERS FOR PAPER / ANALYSIS")
    print("="*50)
    print("UPPER BOUND (Delta+):")
    print(f"  sigma_base = {params_plus[0]:.5f}")
    print(f"  T_limit    = {params_plus[1]:.5f}")
    print(f"  exponent   = {params_plus[2]:.5f}")
    print("-" * 30)
    print("LOWER BOUND (Delta-):")
    print(f"  sigma_base = {params_minus[0]:.5f}")
    print(f"  T_limit    = {params_minus[1]:.5f}")
    print(f"  exponent   = {params_minus[2]:.5f}")
    print("="*50)