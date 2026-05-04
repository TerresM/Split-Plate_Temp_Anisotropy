import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

def rdf_fit_diagnostic_plot_mcmc(ds, block, block_time, result_ds, samples, var_x_array, var_y_array, save_dir=None, safe_time=None):
    """
    Diagnostic plot for the 3D Hierarchical MCMC fitting with Hardware Error Propagation.
    """
    # 1. Extract 3D MCMC Parameters
    a_true_16, alpha_true, a_true_84 = np.percentile(samples[:, 0], [16, 50, 84])
    sig_plus_true = a_true_84 - alpha_true
    sig_minus_true = alpha_true - a_true_16
    
    a_x_16, alpha_eff_x, a_x_84 = np.percentile(samples[:, 1], [16, 50, 84])
    a_y_16, alpha_eff_y, a_y_84 = np.percentile(samples[:, 2], [16, 50, 84])
    
    chi_sq = result_ds.chi_sq_joint.data.item()

    # 2. Extract Data
    vx = block.vx.data.flatten()
    vy = block.vy.data.flatten()
    vz_bx = block.vzbx.data.flatten()
    vz_by = block.vzby.data.flatten()
    
    # Use the EXACT mask that the MCMC used!
    valid_idx = block.mask.data.flatten()

    vdf = block.rdf.data
    normed_vdf = (vdf / np.nanmax(vdf, axis=1)[:, None]).flatten()

    # 3. Setup Spectrogram
    init_time = ds.epoch.data
    vel_mid_copy = ds.vel_mid.data.copy()
    row_mins = np.nanmin(vel_mid_copy, axis=1)
    inds = np.where(np.isnan(vel_mid_copy))
    vel_mid_copy[inds] = row_mins[inds[0]]

    rdf = ds.rdf.data.copy()
    rdf[np.isnan(ds.vel_mid.data)] = np.nan
    normed_rdf = rdf / np.nanmax(rdf, axis=1)[:, None]

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True, sharex=False)

    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(color='none')
    new_cmap = colors.LinearSegmentedColormap.from_list('darker_reds', cmap(np.linspace(0.2, 0.9, 256)))
    new_cmap.set_under("#dde1e8")
    norm = LogNorm(vmin=5e-1, vmax=1)

    # --- PANEL 1: Spectrogram ---
    pm = ax[0].pcolormesh(init_time, vel_mid_copy.T, normed_rdf.T, norm=norm, cmap=new_cmap)
    ax[0].axvline(block_time, color='red', linestyle='--', alpha=0.7)
    
    t_start = block.time.data[0]
    t_end = block.time.data[-1]
    ax[0].axvspan(t_start, t_end, color='red', alpha=0.3, zorder=10)

    ax[0].set_ylabel('$v_z$ [km/s]', fontsize=12)
    ax[0].set_title(f'Hardware Propagated MCMC Fit | {str(block_time)[:20]} \n $T_\\perp/T_\\parallel = {alpha_true:.2f}^{{+{sig_plus_true:.2f}}}_{{-{sig_minus_true:.2f}}}, \\chi^{2}_{{red}} = {chi_sq:.2f}$', fontsize=16, fontweight='bold')
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # --- Calculate Theoretical Physics Lines ---
    bz = result_ds.Nz.data.item() 
    
    def calc_slope(alpha, bz):
        return ((1.0 - alpha) * bz) / (1.0 - (1.0 - alpha) * (1.0 - bz**2))
    
    m_true = calc_slope(alpha_true, bz)
    m_eff_x = calc_slope(alpha_eff_x, bz)
    m_eff_y = calc_slope(alpha_eff_y, bz)

    # --- NEW: Optimal Intercepts using Inverse Variance Weighting ---
    W_x = 1.0 / var_x_array
    W_y = 1.0 / var_y_array
    
    b_eff_x = np.sum(W_x * (vx[valid_idx] - m_eff_x * vz_bx[valid_idx])) / np.sum(W_x)
    b_eff_y = np.sum(W_y * (vy[valid_idx] - m_eff_y * vz_by[valid_idx])) / np.sum(W_y)
    
    b_true_x = np.sum(W_x * (vx[valid_idx] - m_true * vz_bx[valid_idx])) / np.sum(W_x)
    b_true_y = np.sum(W_y * (vy[valid_idx] - m_true * vz_by[valid_idx])) / np.sum(W_y)

    xx_range = np.linspace(np.nanmin(vz_bx[valid_idx]), np.nanmax(vz_bx[valid_idx]), 100)
    xy_range = np.linspace(np.nanmin(vz_by[valid_idx]), np.nanmax(vz_by[valid_idx]), 100)

    # --- PANEL 2: Vx vs Vz Slice ---
    # 1. Plot Hardware Error Bars mapped directly from the var_array
    ax[1].errorbar(vz_bx[valid_idx], vx[valid_idx], yerr=np.sqrt(var_x_array), 
                   fmt='none', ecolor='gray', alpha=0.5, elinewidth=1, capsize=2, zorder=1)
    
    # 2. Plot Valid Core Scatter
    ax[1].scatter(vz_bx[valid_idx], vx[valid_idx], c=normed_vdf[valid_idx], norm=norm, 
                  cmap=new_cmap, edgecolors='k', linewidth=1.0, s=50, zorder=3)
    
    # 3. Plot Ignored Background Scatter
    mask_low = (~valid_idx) & (normed_vdf > 0.0) & np.isfinite(vx) & np.isfinite(vz_bx)
    ax[1].scatter(vz_bx[mask_low], vx[mask_low], c='white', edgecolors='grey', 
                  linewidth=1.0, linestyle='-', s=30, zorder=2, alpha=0.5)

    ax[1].plot(xx_range, m_eff_x * xx_range + b_eff_x, 'b--', lw=2, label=f'Effective X Fit ($\\alpha_{{eff,x}}={alpha_eff_x:.2f}$)', zorder=4)
    ax[1].plot(xx_range, m_true * xx_range + b_true_x, 'k-', lw=3, label=f'True Plasma ($\\alpha_{{true}}={alpha_true:.2f}$)', zorder=5)

    ax[1].set_ylabel('$v_x$ [km/s]', fontsize=12)
    ax[1].set_xlabel('$v_z \cdot b_x$ [nT (km/s)]', fontsize=12)
    ax[1].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    ax[1].grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 3: Vy vs Vz Slice ---
    ax[2].errorbar(vz_by[valid_idx], vy[valid_idx], yerr=np.sqrt(var_y_array), 
                   fmt='none', ecolor='gray', alpha=0.5, elinewidth=1, capsize=2, zorder=1)

    ax[2].scatter(vz_by[valid_idx], vy[valid_idx], c=normed_vdf[valid_idx], norm=norm, 
                  cmap=new_cmap, edgecolors='k', linewidth=1.0, s=50, zorder=3)
                  
    ax[2].scatter(vz_by[mask_low], vy[mask_low], c='white', edgecolors='grey', 
                  linewidth=1.0, linestyle='-', s=30, zorder=2, alpha=0.5)

    ax[2].plot(xy_range, m_eff_y * xy_range + b_eff_y, 'r--', lw=2, label=f'Effective Y Fit ($\\alpha_{{eff,y}}={alpha_eff_y:.2f}$)', zorder=4)
    ax[2].plot(xy_range, m_true * xy_range + b_true_y, 'k-', lw=3, label=f'True Plasma ($\\alpha_{{true}}={alpha_true:.2f}$)', zorder=5)

    ax[2].set_ylabel('$v_y$ [km/s]', fontsize=12)
    ax[2].set_xlabel('$v_z \cdot b_y$ [nT (km/s)]', fontsize=12)
    ax[2].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    ax[2].grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(pm, ax=ax[:], location='right', aspect=40, pad=0.02)
    cbar.set_label('Normalized Phase Space Density', fontsize=12)
    
    # Save or Show Logic
    if save_dir and safe_time:
        plt.savefig(f'{save_dir}/fitting_{safe_time}.png', dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close(fig)

def rdf_fit_diagnostic_plot_mcmc_5D(ds, block, block_time, result_ds, samples):
    """
    Diagnostic plot for the 5D Hierarchical MCMC fitting.
    Plots data points with dynamic error bars derived from the MCMC log_sigma parameters,
    and overlays the True physical anisotropy line and the Effective instrument lines.
    """
    # 1. Extract MCMC Parameters and Asymmetric Uncertainties from the samples array
    a_true_16, alpha_true, a_true_84 = np.percentile(samples[:, 0], [16, 50, 84])
    sig_plus_true = a_true_84 - alpha_true
    sig_minus_true = alpha_true - a_true_16
    
    a_x_16, alpha_eff_x, a_x_84 = np.percentile(samples[:, 1], [16, 50, 84])
    sig_plus_x = a_x_84 - alpha_eff_x
    sig_minus_x = alpha_eff_x - a_x_16

    a_y_16, alpha_eff_y, a_y_84 = np.percentile(samples[:, 2], [16, 50, 84])
    sig_plus_y = a_y_84 - alpha_eff_y
    sig_minus_y = alpha_eff_y - a_y_16

    sig_x = np.exp(np.median(samples[:, 3])) # Convert ln(sigma) back to standard deviation
    sig_y = np.exp(np.median(samples[:, 4]))
    
    # Extract the Reduced Chi-Squared from the dataset
    chi_sq = result_ds.chi_sq_joint.data.item()

    # 2. Extract Data (Flattening arrays to handle potential multi-timestamp blocks)
    vx = block.vx.data.flatten()
    vy = block.vy.data.flatten()
    vz_bx = block.vzbx.data.flatten()
    vz_by = block.vzby.data.flatten()
    
    # Calculate relative weights just like the solver did
    vdf = block.rdf.data
    normed_vdf = (vdf / np.nanmax(vdf, axis=1)[:, None]).flatten()

    # 3. Get the values needed for the 2D pcolormesh (Top Panel)
    init_time = ds.epoch.data
    vel_mid_copy = ds.vel_mid.data.copy()
    row_mins = np.nanmin(vel_mid_copy, axis=1)
    inds = np.where(np.isnan(vel_mid_copy))
    vel_mid_copy[inds] = row_mins[inds[0]]

    rdf = ds.rdf.data.copy()
    rdf[np.isnan(ds.vel_mid.data)] = np.nan
    normed_rdf = rdf / np.nanmax(rdf, axis=1)[:, None]

    # --- Plotting Setup ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True, sharex=False)

    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(color='none')
    new_cmap = colors.LinearSegmentedColormap.from_list('darker_reds', cmap(np.linspace(0.2, 0.9, 256)))
    new_cmap.set_under("#dde1e8")
    norm = LogNorm(vmin=5e-1, vmax=1)

    # --- PANEL 1: Spectrogram ---
    pm = ax[0].pcolormesh(init_time, vel_mid_copy.T, normed_rdf.T, norm=norm, cmap=new_cmap)
    ax[0].axvline(block_time, color='red', linestyle='--', alpha=0.7)
    
    t_start = block.time.data[0]
    t_end = block.time.data[-1]
    ax[0].axvspan(t_start, t_end, color='red', alpha=0.3, zorder=10)

    ax[0].set_ylabel('$v_z$ [km/s]', fontsize=12)
    ax[0].set_title(f'Hierarchical MCMC Fit | {str(block_time)[:20]} \n $T_\\perp/T_\\parallel = {alpha_true:.2f}^{{+{sig_plus_true:.2f}}}_{{-{sig_minus_true:.2f}}}, \\chi^{2} = {chi_sq:.2f}$', fontsize=16, fontweight='bold')
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # --- Calculate Dynamic Error Bars (sigma / sqrt(w)) ---
    # Avoid divide-by-zero for perfectly zero weights
    safe_weights = np.maximum(normed_vdf, 1e-6)
    err_x = sig_x / np.sqrt(safe_weights)
    err_y = sig_y / np.sqrt(safe_weights)

    # Masks for plotting
    mask_high = (normed_vdf >= 0.5) & np.isfinite(vx) & np.isfinite(vz_bx)
    mask_low = (normed_vdf < 0.5) & (normed_vdf > 0.0) & np.isfinite(vx) & np.isfinite(vz_bx)

    # --- Calculate Theoretical Physics Lines ---
    # FIX: Use the unit vector Nz, not the raw magnetic field Bz!
    bz = result_ds.Nz.data.item() 
    
    def calc_slope(alpha, bz):
        return ((1.0 - alpha) * bz) / (1.0 - (1.0 - alpha) * (1.0 - bz**2))
    
    m_true = calc_slope(alpha_true, bz)
    m_eff_x = calc_slope(alpha_eff_x, bz)
    m_eff_y = calc_slope(alpha_eff_y, bz)

    # Calculate the optimal intercepts for the plots just like the MCMC did
    sum_w = np.sum(normed_vdf[mask_high])
    
    # Effective Intercepts
    b_eff_x = np.sum(normed_vdf[mask_high] * (vx[mask_high] - m_eff_x * vz_bx[mask_high])) / sum_w
    b_eff_y = np.sum(normed_vdf[mask_high] * (vy[mask_high] - m_eff_y * vz_by[mask_high])) / sum_w
    
    # True Intercepts
    b_true_x = np.sum(normed_vdf[mask_high] * (vx[mask_high] - m_true * vz_bx[mask_high])) / sum_w
    b_true_y = np.sum(normed_vdf[mask_high] * (vy[mask_high] - m_true * vz_by[mask_high])) / sum_w

    xx_range = np.linspace(np.nanmin(vz_bx[mask_high]), np.nanmax(vz_bx[mask_high]), 100)
    xy_range = np.linspace(np.nanmin(vz_by[mask_high]), np.nanmax(vz_by[mask_high]), 100)

    # --- PANEL 2: Vx vs Vz Slice ---
    # Plot Error Bars
    ax[1].errorbar(vz_bx[mask_high], vx[mask_high], yerr=err_x[mask_high], 
                   fmt='none', ecolor='gray', alpha=0.5, elinewidth=1, capsize=2, zorder=1)
    
    # Plot Scatter
    ax[1].scatter(vz_bx[mask_high], vx[mask_high], c=normed_vdf[mask_high], norm=norm, 
                  cmap=new_cmap, edgecolors='k', linewidth=1.0, s=50, zorder=3)
    ax[1].scatter(vz_bx[mask_low], vx[mask_low], c='white', edgecolors='grey', 
                  linewidth=1.0, linestyle='-', s=30, zorder=2, alpha=0.5)

    # Plot True and Effective Lines (NOW WITH + b_eff_x AND + b_true_x)
    ax[1].plot(xx_range, m_eff_x * xx_range + b_eff_x, 'b--', lw=2, label=f'Effective X Fit ($\\alpha_{{eff,x}}={alpha_eff_x:.2f}$)', zorder=4)
    ax[1].plot(xx_range, m_true * xx_range + b_true_x, 'k-', lw=3, label=f'True Plasma ($\\alpha_{{true}}={alpha_true:.2f}$)', zorder=5)

    ax[1].set_ylabel('$v_x$ [km/s]', fontsize=12)
    ax[1].set_xlabel('$v_z \cdot b_x$ [nT (km/s)]', fontsize=12)
    ax[1].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    ax[1].grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 3: Vy vs Vz Slice ---
    # Plot Error Bars
    ax[2].errorbar(vz_by[mask_high], vy[mask_high], yerr=err_y[mask_high], 
                   fmt='none', ecolor='gray', alpha=0.5, elinewidth=1, capsize=2, zorder=1)

    # Plot Scatter
    ax[2].scatter(vz_by[mask_high], vy[mask_high], c=normed_vdf[mask_high], norm=norm, 
                  cmap=new_cmap, edgecolors='k', linewidth=1.0, s=50, zorder=3)
    ax[2].scatter(vz_by[mask_low], vy[mask_low], c='white', edgecolors='grey', 
                  linewidth=1.0, linestyle='-', s=30, zorder=2, alpha=0.5)

    # Plot True and Effective Lines (NOW WITH + b_eff_y AND + b_true_y)
    ax[2].plot(xy_range, m_eff_y * xy_range + b_eff_y, 'r--', lw=2, label=f'Effective Y Fit ($\\alpha_{{eff,y}}={alpha_eff_y:.2f}$)', zorder=4)
    ax[2].plot(xy_range, m_true * xy_range + b_true_y, 'k-', lw=3, label=f'True Plasma ($\\alpha_{{true}}={alpha_true:.2f}$)', zorder=5)

    ax[2].set_ylabel('$v_y$ [km/s]', fontsize=12)
    ax[2].set_xlabel('$v_z \cdot b_y$ [nT (km/s)]', fontsize=12)
    ax[2].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    ax[2].grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(pm, ax=ax[:], location='right', aspect=40, pad=0.02)
    cbar.set_label('Normalized Phase Space Density', fontsize=12)
    safe_time = str(block_time).replace(':', '').replace('-', '').replace('T', '_')
    date_folder = str(block_time)[:10]
    plt.savefig(f'diagnostic_plots/{date_folder}/fitting_{safe_time}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def rdf_fit_diagnostic_plot_for_groups(init_ds, block, block_time, result_ds):
    # Extract the points that are being fit.
    vx  = block.vx.data
    vy  = block.vy.data

    vz_bx = block.vzbx.data
    vz_by = block.vzby.data

    vdf = block.rdf.data
    normed_vdf = vdf/np.nanmax(vdf, axis=1)[:,None]

    # Get the values needed for the 2D pcolormesh.
    init_time = init_ds.epoch.data

    row_mins = (np.nanmin(init_ds.vel_mid.data.copy(), axis=1))

    vel_mid_copy = init_ds.vel_mid.data.copy()
    inds = np.where(np.isnan(vel_mid_copy))

    vel_mid_copy[inds] = row_mins[inds[0]]

    rdf = init_ds.rdf.data.copy()
    rdf[np.isnan(init_ds.vel_mid.data)] = np.nan

    normed_rdf = rdf/np.nanmax(rdf, axis=1)[:,None]

    # --- 3. Plotting ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=False)

    # Define common styling
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(color='none')

    new_cmap = cmap(np.linspace(0.2, 0.9, 256))
    new_cmap = colors.LinearSegmentedColormap.from_list('darker_reds', new_cmap)

    cmap = new_cmap

    cmap.set_under("#dde1e8")

    v_min, v_max = 5e-1, 1   # Fix these to keep colors consistent across panels
    norm = LogNorm(vmin=v_min, vmax=v_max)

    # --- PANEL 1: Spectrogram (pcolormesh is better than scatter here) ---
    # Note: transposing .rdf to match (y, x) expectation of pcolormesh usually

    pm = ax[0].pcolormesh(init_time, vel_mid_copy.T, normed_rdf.T, norm=norm, cmap=cmap)

    # Add the shaded region for the slice
    # Convert datetime64 to matplotlib float date for width calculation if needed
    ax[0].axvline(block_time, color='red', linestyle='--', alpha=0.7)
    # Create a small window around the slice for visibility (e.g., +/- 1 min)
    t_start = block.time.data[0]
    t_end = block.time.data[-1]
    ax[0].axvspan(t_start, t_end, color='red', alpha=0.3, zorder=10)

    # ax[0].set_yscale('log')
    ax[0].set_ylabel('$v_z$ [km/s]', fontsize=12)
    ax[0].set_title(f'SPC Normalized Reduced Distribution Function | {str(block_time)[:20]} \n $T_\\perp/T_\\parallel = {result_ds.T_ani_avg.data:.2f}\\pm{result_ds.T_ani_avg_uncert.data:.2f}$', fontsize=14)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # --- PANEL 2: Vx vs Vz Slice ---
    mask_low = normed_vdf < 0.5
    mask_high = ~mask_low
    
    # sc1 = ax[1].scatter(vz_bx, vx, c=normed_vdf, norm=norm, cmap=cmap, edgecolors='k', linewidth=0.5, s=50)
    sc1 = ax[1].scatter(vz_bx[mask_high], vx[mask_high], 
                        c=normed_vdf[mask_high], norm=norm, cmap=cmap, 
                        edgecolors='k', linewidth=1.0, s=50, zorder=3)

    # Plot Low Value Points (White Face, Dotted Outline)
    # linestyle=':' makes the edge dotted. c='white' fills the face.
    ax[1].scatter(vz_bx[mask_low], vx[mask_low], 
                  c='white', edgecolors='grey', linewidth=1.0, linestyle='-', s=50, zorder=0)
    
    # Plot Fit Line
    xx_range = np.linspace(np.nanmin(vz_bx), np.nanmax(vz_bx), 100)
    xy_range = np.linspace(np.nanmin(vz_by), np.nanmax(vz_by), 100)


    fit_line_x = result_ds.slope_chi.data * xx_range + result_ds.intercept_chi.data
    fit_line_y = result_ds.slope_psi.data * xy_range + result_ds.intercept_psi.data

    ax[1].plot(xx_range, fit_line_x, 'r-', lw=2, label=f'$\\chi$ = {result_ds.slope_chi.data:.2f}, $T_\\perp/T_\\parallel$ = {result_ds.T_ani_chi.data:.2f}', zorder=0)

    ax[1].set_ylabel('$v_x$ [km/s]', fontsize=12)
    ax[1].set_xlabel('$v_z \cdot b_x$ [nT (km/s)]', fontsize=12)
    ax[1].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8)
    ax[1].grid(True, which='both', linestyle='--', alpha=0.5)


    # --- PANEL 3: Vy vs Vz Slice ---
    # sc2 = ax[2].scatter(vz_by, vy, c=normed_vdf, norm=norm, cmap=cmap, edgecolors='k', linewidth=0.5, s=50)
    sc2 = ax[2].scatter(vz_by[mask_high], vy[mask_high], 
                        c=normed_vdf[mask_high], norm=norm, cmap=cmap, 
                        edgecolors='k', linewidth=1.0, s=50, zorder=3)
    
    # Low Value
    ax[2].scatter(vz_by[mask_low], vy[mask_low], 
                  c='white', edgecolors='grey', linewidth=1.0, linestyle='-', s=50, zorder=0)
        
    
    
    # Plot Fit Line
    ax[2].plot(xy_range, fit_line_y, 'r-', lw=2, label=f'$\\psi$ = {result_ds.slope_psi.data:.2f}, $T_\\perp/T_\\parallel$ = {result_ds.T_ani_psi.data:.2f}', zorder=0)

    ax[2].set_ylabel('$v_y$ [km/s]', fontsize=12)
    ax[2].set_xlabel('$v_z \cdot b_y$ [nT (km/s)]', fontsize=12)
    ax[2].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8)
    ax[2].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Common Limits & Colorbar ---
    # for i in [1, 2]:
    #     ax[i].set_xlim([200, 800])
    #     ax[i].set_ylim([-100, 100])

    # Add a single colorbar for the whole figure (or per panel)
    cbar = fig.colorbar(pm, ax=ax[:], location='right', aspect=40, pad=0.02)
    cbar.set_label('Normalized Phase Space Density', fontsize=12)
    fig.align_ylabels(ax[:])
    plt.show()
    input()


def rdf_fit_diagnostic_plot_individual(init_ds, vx, vy, vz_bx, vz_by, rdf, fit_time, result_ds):
    # Extract the points that are being fit.
    vx  = vx
    vy  = vy

    vz_bx = vz_bx
    vz_by = vz_by

    vdf = rdf
    normed_vdf = vdf/np.nanmax(vdf)

    # Get the values needed for the 2D pcolormesh.
    init_time = init_ds.epoch.data

    row_mins = (np.nanmin(init_ds.vel_mid.data.copy(), axis=1))

    vel_mid_copy = init_ds.vel_mid.data.copy()
    inds = np.where(np.isnan(vel_mid_copy))

    vel_mid_copy[inds] = row_mins[inds[0]]

    rdf = init_ds.rdf.data.copy()
    rdf[np.isnan(init_ds.vel_mid.data)] = np.nan

    normed_rdf = rdf/np.nanmax(rdf, axis=1)[:,None]

    # --- 3. Plotting ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=False)

    # Define common styling
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(color='none')

    new_cmap = cmap(np.linspace(0.2, 0.9, 256))
    new_cmap = colors.LinearSegmentedColormap.from_list('darker_reds', new_cmap)

    cmap = new_cmap

    cmap.set_under("#dde1e8")

    v_min, v_max = 3e-1, 1   # Fix these to keep colors consistent across panels
    norm = LogNorm(vmin=v_min, vmax=v_max)

    # --- PANEL 1: Spectrogram (pcolormesh is better than scatter here) ---
    # Note: transposing .rdf to match (y, x) expectation of pcolormesh usually

    pm = ax[0].pcolormesh(init_time, vel_mid_copy.T, normed_rdf.T, norm=norm, cmap=cmap, rasterized=True)

    # Add the shaded region for the slice
    # Convert datetime64 to matplotlib float date for width calculation if needed
    ax[0].axvline(fit_time, color='red', linestyle='--', alpha=0.7, linewidth=3)

    # ax[0].set_yscale('log')
    ax[0].set_ylabel('$v_z$ [km/s]', fontsize=12)
    # ax[0].set_title('SPC Normalized Reduced Distribution Function', fontsize=14)
    ax[0].set_title(f'SPC Normalized Reduced Distribution Function | {str(fit_time)[:20]} \n $T_\\perp/T_\\parallel = {result_ds.T_ani_avg.data:.2f}\\pm{result_ds.T_ani_avg_uncert.data:.2f}$', fontsize=14)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


    # 1. Create Masks
    mask_low = normed_vdf < 0.35
    mask_high = ~mask_low

    # --- PANEL 2: Vx vs Vz Slice ---
    # sc1 = ax[1].scatter(vz_bx, vx, c=normed_vdf, norm=norm, cmap=cmap, edgecolors='k', linewidth=0.5, s=50)
    sc1 = ax[1].scatter(vz_bx[mask_high], vx[mask_high], 
                        c=normed_vdf[mask_high], norm=norm, cmap=cmap, 
                        edgecolors='k', linewidth=1.0, s=50, zorder=3)

    # 3. Plot Low Value Points (White Face, Dotted Outline)
    # linestyle=':' makes the edge dotted. c='white' fills the face.
    ax[1].scatter(vz_bx[mask_low], vx[mask_low], 
                  c='white', edgecolors='grey', linewidth=1.0, linestyle='-', s=50, zorder=2)
    
    # Plot Fit Line
    xx_range = np.linspace(np.nanmin(vz_bx), np.nanmax(vz_bx), 100)
    xy_range = np.linspace(np.nanmin(vz_by), np.nanmax(vz_by), 100)


    fit_line_x = result_ds.slope_chi.data * xx_range + result_ds.intercept_chi.data
    fit_line_y = result_ds.slope_psi.data * xy_range + result_ds.intercept_psi.data

    ax[1].plot(xx_range, fit_line_x, 'r-', lw=2, label=f'$\\chi$ = {result_ds.slope_chi.data:.2f}, $T_\\perp/T_\\parallel$ = {result_ds.T_ani_chi.data:.2f}', zorder=0)

    ax[1].set_ylabel('$v_x$ [km/s]', fontsize=12)
    ax[1].set_xlabel('$v_z \cdot b_x$ [nT (km/s)]', fontsize=12)
    ax[1].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8)
    ax[1].grid(True, which='both', linestyle='--', alpha=0.5)


    # --- PANEL 3: Vy vs Vz Slice ---
    # sc2 = ax[2].scatter(vz_by, vy, c=normed_vdf, norm=norm, cmap=cmap, edgecolors='k', linewidth=0.5, s=50)
    sc2 = ax[2].scatter(vz_by[mask_high], vy[mask_high], 
                        c=normed_vdf[mask_high], norm=norm, cmap=cmap, 
                        edgecolors='k', linewidth=1.0, s=50, zorder=3)
    
    # Low Value
    ax[2].scatter(vz_by[mask_low], vy[mask_low], 
                  c='white', edgecolors='grey', linewidth=1.0, linestyle='-', s=50, zorder=2)
    
    # Plot Fit Line
    ax[2].plot(xy_range, fit_line_y, 'r-', lw=2, label=f'$\\psi$ = {result_ds.slope_psi.data:.2f}, $T_\\perp/T_\\parallel$ = {result_ds.T_ani_psi.data:.2f}', zorder=0)

    ax[2].set_ylabel('$v_y$ [km/s]', fontsize=12)
    ax[2].set_xlabel('$v_z \cdot b_y$ [nT (km/s)]', fontsize=12)
    ax[2].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8)
    ax[2].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Common Limits & Colorbar ---
    # for i in [1, 2]:
    #     ax[i].set_xlim([200, 800])
    #     ax[i].set_ylim([-100, 100])

    # Add a single colorbar for the whole figure (or per panel)
    cbar = fig.colorbar(pm, ax=ax[:], location='right', aspect=40, pad=0.02)
    cbar.set_label('Normalized Phase Space Density', fontsize=12)
    fig.align_ylabels(ax[:])

    plt.show()
    input()