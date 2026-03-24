import os
import sys
import glob
import cdflib.xarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.constants as c
import astropy.units as u

import warnings
warnings.filterwarnings("ignore")

import matplotlib.colors as colors

def calculate_plasma_beta(ds):
    """
    Calculates scalar proton beta from standard SPC variables.
    Assumes standard PSP units: 
      - Density (np) in cm^-3
      - Thermal Speed (wp) in km/s
      - B-field (b_vec) in nT
    """
    # 1. Constants
    m_p = c.m_p.value       # Proton mass [kg]
    k_B = c.k_B.value       # Boltzmann constant [J/K]
    mu_0 = c.mu0.value      # Vacuum permeability [T^2 m^3 / J]

    # 2. Load Variables (and handle potential naming variations)
    # Prefer Moment data, fall back to Fit if necessary
    if 'np_moment' in ds:
        n_p = ds['np_moment'].data # cm^-3
        w_p = ds['wp_moment'].data # km/s
    else:
        n_p = ds['np_fit'].data
        w_p = ds['wp_fit'].data

    # Magnetic Field Magnitude
    Bx = ds['Bx'].data
    By = ds['By'].data
    Bz = ds['Bz'].data
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2) # nT

    # 3. Unit Conversions to SI
    n_p_SI = n_p * 1e6          # cm^-3 -> m^-3
    w_p_SI = w_p * 1e3          # km/s  -> m/s
    B_SI   = B_mag * 1e-9       # nT    -> T

    # 4. Calculate Temperature (Scalar)
    # T = (m * w^2) / (2 * k_B)
    # Note: This is scalar temperature. For total beta, this is standard.
    T_p_SI = (m_p * w_p_SI**2) / (2 * k_B)

    # 5. Calculate Pressures
    P_thermal = n_p_SI * k_B * T_p_SI
    P_magnetic = (B_SI**2) / (2 * mu_0)

    # 6. Calculate Beta
    beta = P_thermal / P_magnetic
    
    return beta

def calculate_parallel_beta_and_anisotropy(ds, DTYPE='fit'):
    """
    Calculates Parallel Beta (beta_par) and extracts Anisotropy.
    
    Implements the geometric correction:
      T_par = T_fit / (Nz^2 + R * (1 - Nz^2))
      
    Where:
      R  = T_perp / T_par (Anisotropy)
      Nz = Bz / |B|       (Z-component of B-field unit vector)
    """
    # --- 1. Constants ---
    m_p = c.m_p.value       # Proton mass [kg]
    k_B = c.k_B.value       # Boltzmann constant [J/K]
    mu_0 = c.mu0.value      # Vacuum permeability [T^2 m^3 / J]

    # --- 2. Load Basic Quantities ---
    # We prioritize the 'fit' variables for T_fit as implied by your formula, 
    # but you can swap for 'moment' if preferred.
    if DTYPE.lower() == 'fit' and 'np_fit' in ds:
        print('Loading in fit data!')
        n_p = ds['np_fit'].data       # cm^-3
        w_p = ds['wp_fit'].data       # km/s (Thermal Speed)
    elif DTYPE.lower() == 'moment' and 'np_moment' in ds:
        print('Loading in moment data!')
        n_p = ds['np_moment'].data
        w_p = ds['wp_moment'].data
    else:
        # Fallback or error if neither exists
        print('Does not exist! \n\n\n')
        return np.array([]), np.array([])

    # Magnetic Field
    Bx = ds['Bx'].data
    By = ds['By'].data
    Bz = ds['Bz'].data
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2) # nT

    # Anisotropy (R)
    # This is the T_perp / T_par value you calculated previously
    R = ds['T_ani_avg'].data

    # --- 3. Unit Conversions ---
    n_p_SI = n_p * 1e6          # cm^-3 -> m^-3
    w_p_SI = w_p * 1e3          # km/s  -> m/s
    B_SI   = B_mag * 1e-9       # nT    -> T

    # --- 4. Calculate T_fit (Scalar Temperature from LOS fit) ---
    # T = (m * w^2) / (2 * k_B)
    T_fit_K = (m_p * w_p_SI**2) / (2 * k_B)

    # --- 5. Calculate Geometry (Nz) ---
    # Avoid division by zero
    Nz = ds['Nz'].data
    
    # --- 6. Calculate T_parallel (The Correction) ---
    # Formula: T_par = T_fit / (Nz^2 + R * (1 - Nz^2))
    # We use R (Anisotropy) from your file.
    
    denom = (Nz**2) + (R * (1 - Nz**2))
    
    # Filter out bad denominators (e.g., if R is NaN or geometry is weird)
    with np.errstate(divide='ignore', invalid='ignore'):
        T_par_K = T_fit_K / denom

    # --- 7. Calculate Parallel Beta ---
    # Beta_par = P_par / P_mag
    # P_par = n * k * T_par
    
    P_par = n_p_SI * k_B * T_par_K
    P_mag = (B_SI**2) / (2 * mu_0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        beta_par = P_par / P_mag

    return beta_par, R

def Hellinger_expression(beta_para, S, beta0, alpha):
    return(1 + S * (beta_para + beta0)**(-alpha))

def add_hellinger_lines(ax, beta_range=(0.01, 100)):
    """
    Overlays the Hellinger et al. (2006) instability thresholds 
    and the Marsch et al. (2004) inverse correlation line.
    """
    x = np.logspace(np.log10(beta_range[0]), np.log10(beta_range[1]), 200)
    
    # --- 1. Instability Thresholds (gamma = 10^-3 Omega_p) ---
    # Form: R = 1 + a / (beta - beta0)^b
    
    # Proton Cyclotron (AIC)
    # Parameters: a=0.64, b=0.40, beta0=0.02
    # Valid for T_perp > T_par (R > 1)
    mask_aic = (x > 0.01)
    y_aic =  Hellinger_expression(x[mask_aic], 0.43, 0.0004, 0.42)
    ax.plot(x[mask_aic], y_aic, 'k-', linewidth=1.5, label='Cyclotron')

    # Mirror Instability
    # Parameters: a=0.77, b=0.76, beta0=0.016
    # Valid for T_perp > T_par (R > 1)
    mask_mirror = (x > 0.016)
    y_mirror = Hellinger_expression(x[mask_mirror], 0.77, 0.016, 0.76)
    ax.plot(x[mask_mirror], y_mirror, 'k--', linewidth=1.5, label='Mirror')

    # Oblique Firehose
    # Parameters: a=1.4, b=1.0, beta0=-0.11 (Using R = 1 - a / ...)
    # Valid for T_perp < T_par (R < 1)
    mask_ofh = (x > 0) # Technically beta + 0.11 > 0
    y_ofh = Hellinger_expression(x[mask_ofh], -1.4, 0.11, 1)
    # Only plot positive anisotropy
    valid_y_ofh = y_ofh > 0
    ax.plot(x[mask_ofh][valid_y_ofh], y_ofh[valid_y_ofh], 'k-.', linewidth=1.5, label='Oblique Firehose')

    # Parallel Firehose
    # Parameters: a=0.47, b=0.53, beta0=0.59
    # Valid for beta > 0.59
    mask_pfh = (x > 0.59)
    y_pfh = Hellinger_expression(x[mask_pfh], -0.47, -0.59, 0.53)
    valid_y_pfh = y_pfh > 0
    ax.plot(x[mask_pfh][valid_y_pfh], y_pfh[valid_y_pfh], 'k:', linewidth=1.5, label='Parallel Firehose')

    # --- 2. Inverse Correlation Line (Marsch et al. 2004) ---
    # R * beta^0.55 = 1.16  =>  R = 1.16 * beta^-0.55
    y_inv = 1.16 * x**(-0.55)
    ax.plot(x, y_inv, color='dimgray', linestyle='-', linewidth=2, alpha=0.4, label=r'$\beta_{\parallel}^{-0.55}$')

def load_and_aggregate(file_list, max_uncertainty=0.5, PRODUCT='fit'):
    """
    Aggregates data and filters out points where uncertainty is too high.
    
    Args:
        max_uncertainty (float): Maximum allowed uncertainty (fractional or absolute).
                                 Points above this are discarded.
    """
    all_beta_par = []
    all_anisotropy = []
    all_uncertainty = []
    all_uncert_tpara = []
    all_chi = []
    all_psi = []

    print(f"Found {len(file_list)} files. Starting aggregation...")

    for f in file_list:
        ds = cdflib.xarray.cdf_to_xarray(f, to_datetime=True)
        
        # 1. Calculate Physics Variables
        beta_par, anisotropy = calculate_parallel_beta_and_anisotropy(ds, DTYPE=PRODUCT)
        
        # 2. Load Uncertainty Variable
        # IMPORTANT: Check your CDF variable names! 
        if 'T_ani_avg_uncert' in ds:
            uncertainty = 100*ds['T_ani_avg_uncert'].data/ds['T_ani_avg'].data

            T_para_uncert = (((1 - ds['Nz']**2) * ds['T_ani_avg_uncert'])/((ds['Nz']**2) + ds['T_ani_avg'] * (1 - ds['Nz']**2)))**2
        else:
            # If no uncertainty var exists, assume 0 (or skip filtering)
            uncertainty = np.zeros_like(anisotropy) 
            T_para_uncert = np.zeros_like(anisotropy) 

        if 'total_max_flag' in ds:
            print(f'Using max flag for {f}')
            data_max = ds['total_max_flag'].values
        else:
            data_max = np.zeros_like(anisotropy)

        chi = ds['T_ani_chi_uncert']/ds['T_ani_chi']
        psi = ds['T_ani_psi_uncert']/ds['T_ani_psi']
        # 3. Filter Valid Data
        valid_mask = (
            np.isfinite(beta_par) & 
            np.isfinite(anisotropy) & 
            np.isfinite(uncertainty) &
            (beta_par > 1e-4) & (beta_par < 1e3) &
            (anisotropy > 1e-2) & (anisotropy < 1e2) & (data_max > 0) #&
            # (uncertainty < max_uncertainty)  # <--- NEW# FILTER
        )
        
        if np.any(valid_mask):
            all_beta_par.append(beta_par[valid_mask])
            all_anisotropy.append(anisotropy[valid_mask])
            all_uncertainty.append(uncertainty[valid_mask])
            all_uncert_tpara.append(T_para_uncert[valid_mask])
            all_chi.append(chi[valid_mask])
            all_psi.append(psi[valid_mask])

    if len(all_beta_par) > 0:
        return np.concatenate(all_beta_par), np.concatenate(all_anisotropy), np.concatenate(all_uncertainty), np.concatenate(all_uncert_tpara), np.concatenate(all_chi), np.concatenate(all_psi)
    else:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

def calculate_histogram_moments(H, x_edges, y_edges):
    """
    Calculates the 1st and 2nd moments (Mean, Variance, Covariance) 
    directly from a 2D Histogram array.
    
    Assumes H is shape (nx, ny) matching the edges.
    Operates in Log10 space for x and y to handle the wide dynamic range.
    """
    # 1. Convert Edges to Log-Space Centers
    # We use log centers because the physical relationship is power-law based
    log_x_edges = np.log10(x_edges)
    log_y_edges = np.log10(y_edges)
    
    # Calculate centers of the bins
    x_centers = (log_x_edges[:-1] + log_x_edges[1:]) / 2
    y_centers = (log_y_edges[:-1] + log_y_edges[1:]) / 2
    
    # Create a 2D Meshgrid of the coordinates
    # Note: 'indexing="ij"' matches the (nx, ny) shape of H from np.histogram2d
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

    # 2. Zeroth Moment (Total Mass)
    M00 = np.sum(H)
    
    if M00 == 0:
        return None # Handle empty histogram

    # 3. First Moments (Center of Mass)
    # Weighted average of X and Y
    mean_log_x = np.sum(X * H) / M00
    mean_log_y = np.sum(Y * H) / M00

    # 4. Second Central Moments (Variance & Covariance)
    # Variance X
    var_log_x = np.sum(((X - mean_log_x)**2) * H) / M00
    # Variance Y
    var_log_y = np.sum(((Y - mean_log_y)**2) * H) / M00
    # Covariance XY
    cov_log_xy = np.sum(((X - mean_log_x) * (Y - mean_log_y)) * H) / M00

    # 5. Derived Parameters
    # Standard Deviations
    std_log_x = np.sqrt(var_log_x)
    std_log_y = np.sqrt(var_log_y)
    
    # Correlation Coefficient (Pearson)
    correlation = cov_log_xy / (std_log_x * std_log_y)
    
    # Tilt Angle (of the primary axis of the distribution)
    # Calculated from the covariance matrix eigenvectors
    # 0.5 * arctan(2*cov / (var_x - var_y))
    theta_rad = 0.5 * np.arctan2(2 * cov_log_xy, (var_log_x - var_log_y))
    theta_deg = np.degrees(theta_rad)

    return {
        'total_counts': M00,
        'mean_log': (mean_log_x, mean_log_y),
        'mean_linear': (10**mean_log_x, 10**mean_log_y), # Geometric Mean
        'variance_log': (var_log_x, var_log_y),
        'std_log': (std_log_x, std_log_y),
        'covariance_log': cov_log_xy,
        'correlation': correlation,
        'tilt_angle_deg': theta_deg
    }
from matplotlib.patches import Ellipse

def add_moment_ellipse(ax, moments):
    """Draws an ellipse representing the standard deviation and orientation."""
    mu_x = moments['mean_linear'][0] # Linear center (Beta)
    mu_y = moments['mean_linear'][1] # Linear center (Anisotropy)
    
    # Width and Height are typically 2 * std_dev for a 1-sigma ellipse
    # But since we are in Log-Log plots, this is tricky to draw as a simple Ellipse patch
    # because a log-symmetric shape is not an ellipse in linear space.
    # A simple crosshair at the mean is often safer and clearer:
    
    # Draw Mean Point
    label_str = f'{str(mu_x)[:4]}, {str(mu_y)[:4]}'
    ax.plot(mu_x, mu_y, color='tab:red', marker='s', markersize=5, label=f'Mean = {label_str}')
    
    # Draw Error bars (Standard Deviation in log space)
    # Upper/Lower bounds in linear space:
    sigma_x_minus = 10**(moments['mean_log'][0] - moments['std_log'][0])
    sigma_x_plus  = 10**(moments['mean_log'][0] + moments['std_log'][0])
    
    sigma_y_minus = 10**(moments['mean_log'][1] - moments['std_log'][1])
    sigma_y_plus  = 10**(moments['mean_log'][1] + moments['std_log'][1])

    # Plot lines
    ax.plot([sigma_x_minus, sigma_x_plus], [mu_y, mu_y], color='tab:red', linestyle='-', alpha=1.0)
    ax.plot([mu_x, mu_x], [sigma_y_minus, sigma_y_plus], color='tab:red', linestyle='-', alpha=1.0)

def make_brazil_plot(beta_vals, ani_vals, NORMALIZE=False, SAVEFILE=None):
    # 3. Plotting
    if len(beta_vals) > 0:
        print("Data exists. Making Plot...")
    else:
        print('No data is loaded. Exiting...')
        return(0)

    levels_rel = [0.1, 0.32] 
    labels_map = {0.1: '90%', 0.32: '68%'}
    linestyles = ['-.', '-']
    # color_list = ['tab:blue', 'cyan']
    colorstyle = ['lightgray', 'white']

    if NORMALIZE is False:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor('w')

        # We typically use Log-Log bins for Beta vs Anisotropy
        # Define bin edges
        beta_bins = np.logspace(np.log10(0.01), np.log10(15), 100)
        ani_bins  = np.logspace(np.log10(0.1), np.log10(10), 100)

        # Calculate Bin Centers for Contour Plotting
        beta_centers = (beta_bins[:-1] + beta_bins[1:]) / 2
        ani_centers  = (ani_bins[:-1] + ani_bins[1:]) / 2
        # cmap = plt.get_cmap('inferno').copy()
        cmap = plt.get_cmap('Blues').copy()

        new_cmap = cmap(np.linspace(0.2, 1.0, 256))
        new_cmap = colors.LinearSegmentedColormap.from_list('darker_blues', new_cmap)

        cmap = new_cmap
        # cmap.set_under("#000000")
        cmap.set_under("#FFFFFF")
        # Create 2D Histogram
        h = ax.hist2d(beta_vals, ani_vals, bins=[beta_bins, ani_bins], 
                        norm=LogNorm(vmin=5), cmap=cmap)
        
        Z_peak_norm = h[0]/np.max(h[0])
        
        cs = ax.contour(beta_centers, ani_centers, Z_peak_norm.T, 
                    levels=levels_rel, colors=colorstyle, linewidths=2, linestyles=linestyles)    

        moments = calculate_histogram_moments(*h[0:3])

        # Colorbar
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Counts', fontsize=18)

        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel(r'Proton Beta ($\beta_{\parallel p}$)', fontsize=18)
        ax.set_ylabel(r'Temperature Anisotropy ($T_\perp / T_\parallel$)', fontsize=18)
        ax.set_title(f'PSP SPC Anisotropy (Encounters 11-23)', fontsize=20)
        
        # # Add "Firehose" and "Mirror" instability thresholds (Optional reference lines)
        # # Mirror: T_perp/T_par = 1 + 1/beta
        # # Firehose: T_perp/T_par = 1 - 2/beta
        # x_ref = np.logspace(-2, 2, 100)
        # ax.plot(x_ref, 1 + 0.7/x_ref, 'k--', linewidth=3, label='Mirror Instability')
        # ax.plot(x_ref, 1 - 1.4/x_ref, 'k:', linewidth=3, label='Firehose Instability')
        add_hellinger_lines(ax)


        # ax.errorbar(*moments['mean_linear'], xerr=moments['variance_log'][0], yerr=moments['variance_log'][1], fmt='o', ls='none', capsize=4, color='black', markerfacecolor='blue', markeredgecolor='blue')
        add_moment_ellipse(ax, moments)

        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()

        if SAVEFILE:
            plt.savefig(f'{SAVEFILE}.png')
        else:
            plt.show()

        return(moments)
    
    if NORMALIZE == True:
        sys.exit()
        # --- Plotting with Normalization ---
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_facecolor('k')

        # 1. Define Bins
        beta_bins = np.logspace(np.log10(0.01), np.log10(100), 100)
        ani_bins  = np.logspace(np.log10(0.1), np.log10(10), 100)

        # 2. Calculate Histogram Manually
        # H returns shape (nx, ny)
        H, xedges, yedges = np.histogram2d(beta_vals, ani_vals, bins=[beta_bins, ani_bins])

        # 3. Calculate Bin Areas (Linear Space)
        # Since bins are log-spaced, the linear width (dx) grows with x.
        # Area = dx * dy
        dx = np.diff(xedges)
        dy = np.diff(yedges)
        
        # Create 2D Area Map: shape (nx, ny) matches H
        bin_areas = np.outer(dx, dy)

        # 4. Normalize: Density = Counts / Area
        density = H / bin_areas

        # Mask zeros so they appear white/transparent in LogNorm plot
        density_masked = np.ma.masked_where(density == 0, density)

        # 5. Plot using Pcolormesh
        # Note: We pass density.T because pcolormesh expects (rows, cols) -> (y, x)
        pcm = ax.pcolormesh(xedges, yedges, density_masked.T, 
                            norm=LogNorm(), cmap='inferno', shading='auto')
        
        # Colorbar with updated label
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label(r'Phase Space Density ($Counts / \Delta\beta \Delta R$)')

        # --- Add Hellinger & Instability Lines ---
        add_hellinger_lines(ax)

        # --- Formatting ---
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Parallel Proton Beta ($\beta_{\parallel p}$)')
        ax.set_ylabel(r'Temperature Anisotropy ($T_\perp / T_\parallel$)')
        ax.set_title(f'PSP SPC Anisotropy vs Parallel Beta\n(Normalized by Bin Size)')
        
        ax.set_xlim(0.01, 100)
        ax.set_ylim(0.1, 10)
        
        ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        if SAVEFILE:
            plt.savefig(f'{SAVEFILE}.png')
        else:
            plt.show()


if __name__ == "__main__":
    # 1. Setup File Path
    # Change this path to where your .cdf files are stored
    # data_dir = './anisotropy_files/threshold_50_all_events/' 
    # data_dir = './anisotropy_files/threshold_35_all_events/' 
    # data_dir = './anisotropy_files/analytic_uncert_map/'
    data_dir = './anisotropy_files/'
    # data_dir = './anisotropy_files/failed_gamma_avg'
    # data_dir = './anisotropy_files/4_sec_all_encounters/'
    search_pattern = os.path.join(data_dir, 'SPC_Anisotropy_*.cdf')
    files = sorted(glob.glob(search_pattern))

    if not files:
        print("No files found! Check your directory path.")
        exit()

    # 2. Aggregate Data
    beta_vals, ani_vals, uncert_vals, Tpara_uncert, chi_vals, psi_vals = load_and_aggregate(files, max_uncertainty=10, PRODUCT='moment')
    beta_vals_mom, ani_vals_mom, uncert_vals_mom, Tpara_uncert_mom, chi_vals_mom, psi_vals_mom = load_and_aggregate(files, max_uncertainty=10, PRODUCT='moment')

    print(f"Total Data Points Loaded: {len(beta_vals)}")

    make_brazil_plot(beta_vals, ani_vals)