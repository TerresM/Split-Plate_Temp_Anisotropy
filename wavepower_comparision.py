import numpy as np
import cdflib.xarray
import matplotlib.pyplot as plt
import xarray as xr

import astropy.constants as c
import astropy.units as u
import pickle
from datetime import datetime

import matplotlib.colors as colors
from matplotlib.colors import LogNorm

from scipy.stats import binned_statistic_2d

plt.ion()

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
    elif DTYPE.lower() == 'corefit' and 'np1_fit' in ds:
        print('Loading in core fit data!')
        n_p = ds['np1_fit'].data       # cm^-3
        w_p = ds['wp1_fit'].data       # km/s (Thermal Speed)
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
    ax.plot(x[mask_ofh][valid_y_ofh], y_ofh[valid_y_ofh], 'k-.', linewidth=1.5, label='Oblique fire hose')

    # Parallel Firehose
    # Parameters: a=0.47, b=0.53, beta0=0.59
    # Valid for beta > 0.59
    mask_pfh = (x > 0.59)
    y_pfh = Hellinger_expression(x[mask_pfh], -0.47, -0.59, 0.53)
    valid_y_pfh = y_pfh > 0
    ax.plot(x[mask_pfh][valid_y_pfh], y_pfh[valid_y_pfh], 'k:', linewidth=1.5, label='Parallel fire hose')

    # --- 2. Inverse Correlation Line (Marsch et al. 2004) ---
    # R * beta^0.55 = 1.16  =>  R = 1.16 * beta^-0.55
    y_inv = 1.16 * x**(-0.55)
    ax.plot(x, y_inv, color='dimgray', linestyle='-', linewidth=2, alpha=0.4, label=r'$\beta_{\parallel}^{-0.55}$')

def make_scatter_brazil_plot(beta_vals, ani_vals, NORMALIZE=False, SAVEFILE=None):
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
        # h = ax.hist2d(beta_vals, ani_vals, bins=[beta_bins, ani_bins], 
        #                 norm=LogNorm(vmin=5), cmap=cmap)
        
        # Z_peak_norm = h[0]/np.max(h[0])
        
        # cs = ax.contour(beta_centers, ani_centers, Z_peak_norm.T, 
        #             levels=levels_rel, colors=colorstyle, linewidths=2, linestyles=linestyles)    

        ax.scatter(beta_vals, ani_vals, marker='.', color='tab:blue', alpha=0.7, edgecolors='k')

        # Colorbar
        # cbar = plt.colorbar(h[3], ax=ax)
        # cbar.set_label('Counts', fontsize=18)

        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel(r'Proton Beta ($\beta_{\parallel p}$)', fontsize=18)
        ax.set_ylabel(r'Temperature Anisotropy ($T_\perp / T_\parallel$)', fontsize=18)
        ax.set_title(f'PSP SPC Anisotropy (Encounters 11-23)', fontsize=20)
        
        add_hellinger_lines(ax)

        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()

def make_contour_brazil_plot(beta_vals, ani_vals, NORMALIZE=False, SAVEFILE=None):
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

        # Colorbar
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Counts', fontsize=18)

        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel(r'Proton Beta ($\beta_{\parallel p}$)', fontsize=18)
        ax.set_ylabel(r'Temperature Anisotropy ($T_\perp / T_\parallel$)', fontsize=18)
        ax.set_title(f'PSP SPC Anisotropy (Encounters 11-23)', fontsize=20)
        
        add_hellinger_lines(ax)

        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()

if __name__ == "__main__":
    # date_list = ['2023-09-21', '2023-09-22', '2023-09-23', '2023-09-24', '2023-09-25', '2023-09-26']

    file_path = 'selected_dates_updated.pkl'

    # Load the list (open file in 'rb' binary read mode)
    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    loaded_list = [date for date_list in loaded_dict.values() for date in date_list]

    cutoff = datetime(2023,1,1) 

    laoded_list = [
        d for d in loaded_list
        if datetime.strptime(d, "%Y-%m-%d") > cutoff
    ]

    xr_wave_list = []
    xr_ani_list = []
    for date in loaded_list:
        try:
            year, month, day = date.split('-')
            xr_wave = cdflib.xarray.cdf_to_xarray(f'./v1.3/{year}/{month}/PSP_wavePower_{date}_v1.3.cdf')
            # xr_ani  = cdflib.xarray.cdf_to_xarray(f'./anisotropy_files/SPC_Anisotropy_{date}_peak_track_bavg__averaged_4s.cdf')
            xr_ani  = cdflib.xarray.cdf_to_xarray(f'./anisotropy_files/SPC_Anisotropy_{date}_peak_track_averaged_4s_Final.cdf')
            # xr_ani  = cdflib.xarray.cdf_to_xarray(f'./anisotropy_files/stat_fit/SPC_Anisotropy_{date}_peak_track_averaged_4s_NEW_3PFIT_statfit.cdf')
            # xr_ani  = cdflib.xarray.cdf_to_xarray(f'./anisotropy_files/SPC_Anisotropy_{date}_peak_track_averaged_4s_60s_field.cdf')
            
            xr_wave = xr_wave.where(xr_wave != -1e31, np.nan)
            xr_ani = xr_ani.where(xr_ani != -1e31, np.nan)

            xr_wave_list.append(xr_wave)
            xr_ani_list.append(xr_ani)
        except:
            print(f'No data found for {date}')

    xr_wave_merge = xr.concat(xr_wave_list, dim='epoch')
    xr_ani_merge  = xr.concat(xr_ani_list, dim='time')

    # We are going to do a 10 point rolling average of xr_wave
    # xr_wave_roll = xr_wave_merge.rolling(epoch=10, center=True).mean()
    xr_wave_resample = xr_wave_merge.resample(epoch='4s').max()

    # xr_wave_int = xr_wave_roll.interp(epoch=xr_ani_merge.time.data)
    xr_wave_int = xr_wave_resample.interp(epoch=xr_ani_merge.time.data)

    # Get the parallel plasma beta
    beta_para, T_ani = calculate_parallel_beta_and_anisotropy(xr_ani_merge, DTYPE='corefit')

    max_flag = (xr_ani_merge.total_max_flag.data > 0)
    
    uncert = (50 * (xr_ani_merge.T_ani_avg_delta_plus.data + xr_ani_merge.T_ani_avg_delta_plus.data))/xr_ani_merge.T_ani_avg.data
    # uncert = (100 * xr_ani_merge.T_ani_avg_uncert.data)/xr_ani_merge.T_ani_avg.data
    uncert[np.isinf(uncert)] = np.nan

    uncert_flag = (uncert < 50) & np.isfinite(uncert)

    t_ani_flag = (T_ani > 0.15) & (T_ani < 9.85)

    max_flag = max_flag & uncert_flag & t_ani_flag

    # Lets try the scatter plot colored by wave power.
    x_data = np.log10(beta_para[max_flag])
    y_data = np.log10(T_ani[max_flag])

    x_bins_log = np.linspace(-2, 1.18, 100)
    y_bins_log = np.linspace(np.log10(0.1), np.log10(10), 100)

    statistic_lh, x_edge, y_edge, binnumber_lh = binned_statistic_2d(
        x=x_data,
        y=y_data,
        values=xr_wave_int.wavePower_LH.data[max_flag],
        statistic='median',
        bins=[x_bins_log, y_bins_log]
    )

    statistic_rh, x_edge, y_edge, binnumber_rh = binned_statistic_2d(
        x=x_data,
        y=y_data,
        values=xr_wave_int.wavePower_RH.data[max_flag],
        statistic='median',
        bins=[x_bins_log, y_bins_log]
    )

    counts, _, _, binnumber2 = binned_statistic_2d(
        x=x_data,
        y=y_data,
        values=xr_wave_int.wavePower_LH.data[max_flag],
        statistic='count',
        bins=[x_bins_log, y_bins_log]
    )

    # Mask out empty bins so they plot as blank space instead of 0
    # statistic_lh_masked = np.ma.masked_where(counts < 10, statistic)
    statistic_lh[counts < 10] = np.nan
    statistic_rh[counts < 10] = np.nan

    statistic_lh_masked = np.ma.masked_invalid(statistic_lh)
    statistic_rh_masked = np.ma.masked_invalid(statistic_rh)

    # Convert bin edges back to linear space for plotting
    X_grid, Y_grid = np.meshgrid(10**x_edge, 10**y_edge)


    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the 2D histogram
    # We transpose the statistic array (.T) because binned_statistic_2d returns it as (Nx, Ny)
    # but pcolormesh expects (Ny, Nx)
    im = ax.pcolormesh(X_grid, Y_grid, statistic_lh_masked.T, 
                    cmap='Blues', norm=LogNorm(), shading='flat')
    
    # Add the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Median LH Wave Power', fontsize=14)

    add_hellinger_lines(ax)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 15)
    ax.set_ylim(1e-1, 10)

    ax.set_xlabel('Proton Beta ($\\beta_{\\parallel p}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Temperature Anisotropy ($T_\\perp / T_\\parallel$)', fontsize=16, fontweight='bold')
    ax.set_title('SPC Temperature Anisotropy vs Wave Power', fontsize=18, fontweight='bold')

    # ==========================================
    # DRAW THE BINNING GRID
    # ==========================================
    # Draw vertical lines for every x bin edge
    ax.vlines(10**x_edge, 
              ymin=10**y_edge[0], ymax=10**y_edge[-1], 
              color='lightgray', linewidth=0.4, alpha=0.6, zorder=1)
    
    # Draw horizontal lines for every y bin edge
    ax.hlines(10**y_edge, 
              xmin=10**x_edge[0], xmax=10**x_edge[-1], 
              color='lightgray', linewidth=0.4, alpha=0.6, zorder=1)

    # ax.grid(True, which='both', linestyle='-', alpha=0.2)
    ax.legend(loc='upper right', fontsize=12)
















    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Plot the 2D histogram
    # We transpose the statistic array (.T) because binned_statistic_2d returns it as (Nx, Ny)
    # but pcolormesh expects (Ny, Nx)
    im = ax.pcolormesh(X_grid, Y_grid, statistic_lh_masked.T, 
                    cmap='Blues', norm=LogNorm(vmin=1e-4, vmax=1e2), shading='flat')
    
    im2 = ax.pcolormesh(X_grid, Y_grid, statistic_rh_masked.T, 
                    cmap='Reds', norm=LogNorm(vmin=1e-4, vmax=1e2), shading='flat')
    


    # Add the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar2 = fig.colorbar(im2, ax=ax)
    cbar.set_label('Median LH Wave Power', fontsize=14, fontweight='bold')
    cbar2.set_label('Median RH Wave Power', fontsize=14, fontweight='bold')

    add_hellinger_lines(ax)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 15)
    ax.set_ylim(1e-1, 10)

    ax.set_xlabel('Proton Beta ($\\beta_{\\parallel p}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Temperature Anisotropy ($T_\\perp / T_\\parallel$)', fontsize=16, fontweight='bold')
    ax.set_title('SPC Anisotropy vs Wave Power', fontsize=18, fontweight='bold')

    # ==========================================
    # DRAW THE BINNING GRID
    # ==========================================
    # Draw vertical lines for every x bin edge
    ax.vlines(10**x_edge, 
              ymin=10**y_edge[0], ymax=10**y_edge[-1], 
              color='lightgray', linewidth=0.4, alpha=0.6, zorder=1)
    
    # Draw horizontal lines for every y bin edge
    ax.hlines(10**y_edge, 
              xmin=10**x_edge[0], xmax=10**x_edge[-1], 
              color='lightgray', linewidth=0.4, alpha=0.6, zorder=1)

    # ax.grid(True, which='both', linestyle='-', alpha=0.2)
    ax.legend(loc='upper right', fontsize=12)

    plt.savefig('./figures/wavepower_brazil_plot.png')



    # I need to get the time index near the LH waves we suspect to be doppler shifted. 
    # Let us get the bin locations.
    y_bins_total = len(y_edge) + 1

    lh_bin_x = binnumber_lh // y_bins_total
    lh_bin_y = binnumber_rh % y_bins_total

    # Some points are mapped outside the domain, therefore, we will drop those. 
    valid_points_mask = (lh_bin_x > 0) & (lh_bin_x < len(x_edge)) & (lh_bin_y > 0) & (lh_bin_y < len(y_edge))

    valid_lh_bin_x = lh_bin_x[valid_points_mask]
    valid_lh_bin_y = lh_bin_y[valid_points_mask]

    times = xr_ani_merge.time.data[max_flag]

    valid_times = times[valid_points_mask]

    anis = 10**(y_edge[valid_lh_bin_y])
    beta = 10**(x_edge[valid_lh_bin_x])


