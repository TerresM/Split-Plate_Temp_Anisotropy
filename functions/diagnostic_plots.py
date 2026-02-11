import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from matplotlib.colors import LogNorm

import matplotlib.colors as colors

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
    # plt.show()
    # input()


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
    ax[0].set_title('SPC Normalized Reduced Distribution Function', fontsize=14)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


    # 1. Create Masks
    mask_low = normed_vdf < 0.5
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

    ax[1].plot(xx_range, fit_line_x, 'r-', lw=2, label=f'Fit Slope: {result_ds.slope_chi.data:.2f}', zorder=0)

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
    ax[2].plot(xy_range, fit_line_y, 'r-', lw=2, label=f'Fit Slope: {result_ds.slope_psi.data:.2f}', zorder=0)

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