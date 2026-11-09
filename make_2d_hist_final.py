import re
import glob
import numpy as np
import pandas as pd
import cdflib.xarray
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import pickle
import xarray as xr
import astropy.constants as c
import sys

from scipy.interpolate import interp1d
from scipy.integrate import quad
from datetime import datetime
from matplotlib.colors import LogNorm
from sunpy.coordinates import spice

plt.ion()


def calculate_parallel_beta_and_anisotropy(ds, DTYPE='fit'):
    """
    Calculates Parallel Beta (beta_par) and extracts Anisotropy.
    Implements the geometric correction: T_par = T_fit / (Nz^2 + R * (1 - Nz^2))
    """
    m_p = c.m_p.value       
    k_B = c.k_B.value       
    mu_0 = c.mu0.value      

    if DTYPE.lower() == 'fit' and 'np_fit' in ds:
        n_p = ds['np_fit'].data       
        w_p = ds['wp_fit'].data       
    elif DTYPE.lower() == 'corefit' and 'np1_fit' in ds:
        n_p = ds['np1_fit'].data       
        w_p = ds['wp1_fit'].data       
    elif DTYPE.lower() == 'moment' and 'np_moment' in ds:
        n_p = ds['np_moment'].data
        w_p = ds['wp_moment'].data
    else:
        print('Data arrays do not exist!')
        return np.array([]), np.array([])

    Bx = ds['Bx'].data
    By = ds['By'].data
    Bz = ds['Bz'].data
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2) 

    R = ds['T_ani_avg'].data

    n_p_SI = n_p * 1e6          
    w_p_SI = w_p * 1e3          
    B_SI   = B_mag * 1e-9       

    T_fit_K = (m_p * w_p_SI**2) / (2 * k_B)
    Nz = ds['Nz'].data
    
    denom = (Nz**2) + (R * (1 - Nz**2))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        T_par_K = T_fit_K / denom

    P_par = n_p_SI * k_B * T_par_K
    P_mag = (B_SI**2) / (2 * mu_0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        beta_par = P_par / P_mag

    return beta_par, R


def Hellinger_expression(beta_para, S, beta0, alpha):
    return (1 + S * (beta_para + beta0)**(-alpha))


def add_hellinger_lines(ax, beta_range=(0.01, 100)):
    """Overlays the Hellinger et al. (2006) and Marsch et al. (2004) limits."""
    x = np.logspace(np.log10(beta_range[0]), np.log10(beta_range[1]), 200)
    
    # Cyclotron
    mask_aic = (x > 0.01)
    y_aic = Hellinger_expression(x[mask_aic], 0.43, 0.0004, 0.42)
    ax.plot(x[mask_aic], y_aic, 'k-', linewidth=1.5, label='Cyclotron')

    # Mirror
    mask_mirror = (x > 0.016)
    y_mirror = Hellinger_expression(x[mask_mirror], 0.77, 0.016, 0.76)
    ax.plot(x[mask_mirror], y_mirror, 'k--', linewidth=1.5, label='Mirror')

    # Oblique Firehose
    mask_ofh = (x > 0)
    y_ofh = Hellinger_expression(x[mask_ofh], -1.4, 0.11, 1)
    valid_y_ofh = y_ofh > 0
    ax.plot(x[mask_ofh][valid_y_ofh], y_ofh[valid_y_ofh], 'k-.', linewidth=1.5, label='Oblique fire hose')

    # Parallel Firehose
    mask_pfh = (x > 0.59)
    y_pfh = Hellinger_expression(x[mask_pfh], -0.47, -0.59, 0.53)
    valid_y_pfh = y_pfh > 0
    ax.plot(x[mask_pfh][valid_y_pfh], y_pfh[valid_y_pfh], 'k:', linewidth=1.5, label='Parallel fire hose')

    # # ---------------------------------------------------------
    # # Hot Dispersion Marginal Stability (Cranmer 2014)
    # # ---------------------------------------------------------
    # n = 5.0/3.0
    # def A_integrand(x_int, beta):
    #     if x_int == 0: return 0.0
    #     return -2 * ((x_int - 1)**2 / x_int**(n + 3)) * np.exp(-((x_int - 1)**2) / (x_int**2 * beta))

    # def B_integrand(x_int, beta):
    #     if x_int == 0: return 0.0
    #     return ((1 - x_int) / x_int**(n + 3)) * np.exp(-((x_int - 1)**2) / (x_int**2 * beta))

    # def C_integrand(x_int, beta):
    #     if x_int == 0: return 0.0
    #     return (1 / x_int**(n + 3)) * np.exp(-((x_int - 1)**2) / (x_int**2 * beta))

    # R_vals = []
    # for b in x:
    #     A_val, _ = quad(A_integrand, 0, 1, args=(b,))
    #     B_val, _ = quad(B_integrand, 0, 1, args=(b,))
    #     C_integral, _ = quad(C_integrand, 0, 1, args=(b,))
        
    #     C_val = C_integral + 1 / (n + 2)
    #     discriminant = B_val**2 - 4 * A_val * C_val
        
    #     if discriminant < 0:
    #         R_vals.append(np.nan)
    #     else:
    #         R_vals.append((-B_val - np.sqrt(discriminant)) / (2 * A_val))
            
    # # Plot the newly derived marginal stability curve (magenta, solid)
    # ax.plot(x, R_vals, color='magenta', linestyle='-', linewidth=2.5, label='Hot Dispersion')



    # ---------------------------------------------------------
    # Marsch et al. 2004 limits (Inline Text Labels)
    # ---------------------------------------------------------
    inline_bbox = dict(facecolor='white', edgecolor='none', alpha=1.0, pad=2)

    # Plot -0.55 line 
    y_inv_55 = 0.92 * x**(-0.55)
    ax.plot(x, y_inv_55, color='chocolate', linestyle='-', linewidth=2, alpha=1.0)
    
    ax.text(10, 0.26, r'$-0.55$', color='chocolate', 
            fontsize=12, ha='center', va='center', rotation=-33,
            bbox=inline_bbox, zorder=6)

    # Plot -0.40 line
    y_inv_40 = 0.92 * x**(-0.40)
    ax.plot(x, y_inv_40, color='chocolate', linestyle='--', linewidth=2, alpha=1.0)
    
    ax.text(6.5, 0.44, r'$-0.40$', color='chocolate', 
            fontsize=12, ha='center', va='center', rotation=-25,
            bbox=inline_bbox, zorder=6)


def calculate_fwhm_bounds(slice_1d, center_idx, bin_centers):
    """Finds the left and right Full Width at Half Maximum (FWHM) bounds for a 1D slice."""
    max_val = slice_1d[center_idx]
    hm = max_val / 2.0
    log_bins = np.log10(bin_centers)
    
    # Trace backwards to find the Left Bound
    left_idx = center_idx
    while left_idx > 0 and np.nan_to_num(slice_1d[left_idx]) > hm:
        left_idx -= 1
        
    x0, x1 = log_bins[left_idx], log_bins[left_idx+1]
    y0, y1 = np.nan_to_num(slice_1d[left_idx]), np.nan_to_num(slice_1d[left_idx+1])
    
    if y1 != y0:
        left_bound_log = x0 + (hm - y0) * (x1 - x0) / (y1 - y0)
    else:
        left_bound_log = x0
        
    # Trace forwards to find the Right Bound
    right_idx = center_idx
    while right_idx < len(slice_1d) - 1 and np.nan_to_num(slice_1d[right_idx]) > hm:
        right_idx += 1
        
    x0, x1 = log_bins[right_idx-1], log_bins[right_idx]
    y0, y1 = np.nan_to_num(slice_1d[right_idx-1]), np.nan_to_num(slice_1d[right_idx])
    
    if y1 != y0:
        right_bound_log = x0 + (hm - y0) * (x1 - x0) / (y1 - y0)
    else:
        right_bound_log = x1
        
    return 10**left_bound_log, 10**right_bound_log


def add_fwhm_crosshair(ax, peak_x, peak_y, x_bounds, y_bounds):
    """Draws the crosshair centered on the peak, spanning the FWHM."""
    outline = [pe.withStroke(linewidth=3, foreground='white')]
    
    # Plot Peak as a prominent star
    ax.plot(peak_x, peak_y, color='deeppink', marker='*', markersize=14, 
            path_effects=outline, zorder=11)

    # Plot FWHM crosshair lines
    ax.plot([x_bounds[0], x_bounds[1]], [peak_y, peak_y], color="deeppink", 
            linestyle='-', alpha=1.0, path_effects=outline, zorder=10)
    ax.plot([peak_x, peak_x], [y_bounds[0], y_bounds[1]], color="deeppink", 
            linestyle='-', alpha=1.0, path_effects=outline, zorder=10)


def plot_brazil_panel(ax, beta_vals, ani_vals, title, NBINS=100, PRINT_MOMENTS=False, SHOW_LEGEND=True, CMAP='Blues'):
    """Generates the Area-Normalized 2D histogram on a specified axis panel."""
    if len(beta_vals) == 0:
        return None

    beta_bins = np.logspace(np.log10(0.01), np.log10(15), NBINS)
    ani_bins  = np.logspace(np.log10(0.1), np.log10(10), NBINS)
    
    # beta_centers = (beta_bins[:-1] + beta_bins[1:]) / 2
    # ani_centers  = (ani_bins[:-1] + ani_bins[1:]) / 2

    beta_centers = np.sqrt(beta_bins[:-1] * beta_bins[1:])
    ani_centers  = np.sqrt(ani_bins[:-1] * ani_bins[1:])

    cmap = plt.get_cmap(CMAP).copy()
    new_cmap = cmap(np.linspace(0.2, 1.0, 256))
    new_cmap = colors.LinearSegmentedColormap.from_list('darker_blues', new_cmap)
    new_cmap.set_under("#FFFFFF")

    # 1. Calculate Raw 2D Histogram
    H, xedges, yedges = np.histogram2d(beta_vals, ani_vals, bins=[beta_bins, ani_bins])
    total_counts = np.sum(H)
    if total_counts == 0: return None

    # 2. Calculate Physical Bin Areas
    db = np.diff(xedges)
    da = np.diff(yedges)
    bin_areas = np.outer(db, da)

    # 3. Normalize by Area (Density = Counts / Area)
    with np.errstate(divide='ignore', invalid='ignore'):
        H_norm = H / (total_counts * bin_areas)
        # Set empty bins to NaN so they render as the background color (white)
        H_norm[H < 5] = np.nan 

    # 4. Plot using pcolormesh
    mesh = ax.pcolormesh(xedges, yedges, H_norm.T, norm=LogNorm(), cmap=new_cmap)

    # 5. Calculate Contours
    flat_H = H.flatten()
    flat_H_norm = H_norm.flatten()
    
    sort_idx = np.argsort(np.nan_to_num(flat_H_norm))[::-1]
    
    sorted_H_norm = flat_H_norm[sort_idx]
    sorted_H = flat_H[sort_idx]
    
    cumsum_H = np.cumsum(sorted_H)
    idx_50 = np.searchsorted(cumsum_H, 0.50 * total_counts)
    idx_75 = np.searchsorted(cumsum_H, 0.75 * total_counts)
    
    level_50 = sorted_H_norm[idx_50]
    level_75 = sorted_H_norm[idx_75]

    ax.contour(beta_centers, ani_centers, H_norm.T, 
               levels=[level_75, level_50], colors=["lightgrey", "white"], 
               linewidths=2, linestyles=['-.', '-'])

    # 6. Find Most Probable Value (True Peak based on Density)
    max_idx = np.unravel_index(np.nanargmax(H_norm), H_norm.shape)
    peak_beta = beta_centers[max_idx[0]]
    peak_ani = ani_centers[max_idx[1]]

    # 7. Extract 1D Density Slices intersecting the Peak
    slice_beta = H_norm[:, max_idx[1]]
    slice_ani = H_norm[max_idx[0], :]
    
    # 8. Calculate FWHM Bounds
    beta_left, beta_right = calculate_fwhm_bounds(slice_beta, max_idx[0], beta_centers)
    ani_bottom, ani_top = calculate_fwhm_bounds(slice_ani, max_idx[1], ani_centers)

    if PRINT_MOMENTS:
        beta_upper_diff = beta_right - peak_beta
        beta_lower_diff = peak_beta - beta_left
        ani_upper_diff = ani_top - peak_ani
        ani_lower_diff = peak_ani - ani_bottom
        
        peak_text = (
            f"$\\beta_{{\\parallel, peak}} = {peak_beta:.3f}^{{+{beta_upper_diff:.3f}}}_{{-{beta_lower_diff:.3f}}}$\n"
            f"$(T_{{\\perp}}/T_{{\\parallel}})_{{peak}} = {peak_ani:.3f}^{{+{ani_upper_diff:.3f}}}_{{-{ani_lower_diff:.3f}}}$"
        )
        
        ax.text(0.95, 0.95, peak_text, transform=ax.transAxes, 
                ha='right', va='top', fontsize=16,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='gray'))
        
        print(f"--- {title} Peak & FWHM Statistics ---")
        print(f"Total Valid Points: {total_counts:,.0f}")
        print(f"Peak Plasma Beta:   {peak_beta:.3f} (FWHM bounds: {beta_left:.3f} to {beta_right:.3f})")
        print(f"Peak Anisotropy:    {peak_ani:.3f} (FWHM bounds: {ani_bottom:.3f} to {ani_top:.3f})")
        print("------------------------------------------")

    add_hellinger_lines(ax)
    add_fwhm_crosshair(ax, peak_beta, peak_ani, (beta_left, beta_right), (ani_bottom, ani_top))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel(r'Parallel Plasma Beta ($\beta_{\parallel p}$)', fontsize=16, fontweight='bold')
    ax.grid(True, which='both', alpha=0.5)
    
    if SHOW_LEGEND:
        ax.legend(ncols=1, loc='lower left', fontsize=16, framealpha=1.0)
    
    return mesh


def get_boundary_function(filename):
        df_model = pd.read_csv(filename, skiprows=1)
        return interp1d(
            df_model['Heliocentric Distance (R_Sun)'], 
            df_model['Solar Wind Speed (km/s)'], 
            kind='linear', 
            bounds_error=False, 
            fill_value="extrapolate"
        )


def plot_composite_classifications(
    r_spc, vsw, beta_para_core, anisotropy, 
    models, threshold_slow_speeds, threshold_fast_speeds,
    max_flag, strict_uncert_flag, valid_anis, NBINS=100
):
    flag_all = (max_flag) & (strict_uncert_flag) & (valid_anis)
    flag_slow = flag_all & (vsw < threshold_slow_speeds)
    flag_fast = flag_all & (vsw > threshold_fast_speeds)
    mask_inter = max_flag & (vsw >= threshold_slow_speeds) & (vsw <= threshold_fast_speeds)
    mask_slow_scatter = max_flag & (vsw < threshold_slow_speeds)
    mask_fast_scatter = max_flag & (vsw > threshold_fast_speeds)

    models.sort(key=lambda x: x[0])
    
    # --- FIGURE SETUP USING A SINGLE GRIDSPEC ---
    fig = plt.figure(figsize=(20, 14))
    
    # 4 columns: Columns 0, 1, 2 hold plots. Column 3 holds the bottom right colorbar.
    gs = fig.add_gridspec(nrows=2, ncols=4, height_ratios=[1, 1.2], 
                          width_ratios=[1, 1, 1, 0.03], hspace=0.3, wspace=0.15)

    # --- TOP PANELS ---
    # Panel (a) spans columns 0 and 1. This guarantees it covers the exact width of panels (c) and (d).
    ax_top = fig.add_subplot(gs[0, 0:2])
    # Panel (b) spans column 2. This guarantees it perfectly aligns over panel (e).
    ax_hist = fig.add_subplot(gs[0, 2])

    min_speed, max_speed = 273, 804
    base_cmap = plt.get_cmap('plasma_r')
    plasma_r_shift = base_cmap(np.linspace(0.2, 1.0, 256))
    cmap = colors.LinearSegmentedColormap.from_list('plasma_r_shift', plasma_r_shift)
    norm = colors.Normalize(vmin=min_speed, vmax=max_speed)

    for speed, filename in models:
        df_model = pd.read_csv(filename, skiprows=1)
        r_model = df_model['Heliocentric Distance (R_Sun)']
        v_model = df_model['Solar Wind Speed (km/s)']
        ax_top.plot(r_model, v_model, color=cmap(norm(speed)), alpha=0.6, linewidth=2.0, zorder=1)

    ax_top.scatter(r_spc[mask_inter], vsw[mask_inter], color='black',  s=2, alpha=0.2, zorder=5, label='Intermediate Wind')
    ax_top.scatter(r_spc[mask_slow_scatter], vsw[mask_slow_scatter], color='tab:blue', s=2, alpha=0.8, zorder=5, label='Slow Wind (< 400 km/s)')
    ax_top.scatter(r_spc[mask_fast_scatter], vsw[mask_fast_scatter], color='crimson',  s=2, alpha=0.8, zorder=5, label='Fast Wind (> 550 km/s)')

    ax_top.set_xlabel(r'Heliocentric Distance ($R_{\odot}$)', fontsize=16, fontweight='bold')
    ax_top.set_ylabel('Solar Wind Speed (km/s)', fontsize=16, fontweight='bold')
    ax_top.set_title('(a) Solar Wind Speed Classifications', fontsize=18, fontweight='bold')
    ax_top.set_xscale('log')
    
    leg = ax_top.legend(loc='upper left', fontsize=16, markerscale=4)
    for lh in leg.legend_handles: 
        lh.set_alpha(1.0)

    ax_top.grid(True, linestyle='--', alpha=0.4)
    ax_top.set_xlim([10, 100])
    ax_top.set_ylim([100, 850])
    ax_top.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    # --- COLORBAR ALIGNMENT MAGIC ---
    # By passing `ax=ax_top`, Matplotlib builds the colorbar INSIDE the bounds of gs[0, 0:2].
    # It shrinks Panel (a) to fit, and the right edge of the colorbar perfectly aligns 
    # with the right edge of gs[0, 0:2], which is exactly aligned with Panel (d) below it!
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    # fraction: Controls thickness of the colorbar
    # pad: Controls the tiny gap between the colorbar and the main plot
    cbar_top = fig.colorbar(sm, ax=ax_top, fraction=0.025, pad=0.015, aspect=30)
    cbar_top.ax.set_title('Isopoly Model\n1 AU Speed (km/s)', fontsize=11, fontweight='bold', pad=10)

    # --- TOP PANEL (b): 1D HISTOGRAM ---
    ani_valid = anisotropy[flag_all]
    ani_s = anisotropy[flag_slow]
    ani_f = anisotropy[flag_fast]

    ani_bins = np.linspace(np.log10(0.1), np.log10(10), NBINS)

    counts_v, log_edges_v = np.histogram(np.log10(ani_valid), bins=ani_bins, density=False)
    counts_s, log_edges_s = np.histogram(np.log10(ani_s), bins=ani_bins, density=False)
    counts_f, log_edges_f = np.histogram(np.log10(ani_f), bins=ani_bins, density=False)

    lin_edges = 10**log_edges_s

    v_points = np.sum(counts_v)
    s_points = np.sum(counts_s)
    f_points = np.sum(counts_f)

    pdf_v = counts_v / (v_points* np.diff(log_edges_v))
    pdf_s = counts_s / (s_points* np.diff(log_edges_s)) # * np.diff(log_edges))
    pdf_f = counts_f / (f_points* np.diff(log_edges_f))

    ax_hist.stairs(pdf_v, lin_edges, fill=True, alpha=0.4, color='grey', edgecolor='k', linewidth=2, label='All')
    ax_hist.stairs(pdf_s, lin_edges, fill=False, alpha=0.4, color='tab:blue', linewidth=2, label='Slow')
    ax_hist.stairs(pdf_f, lin_edges, fill=False, alpha=0.4, color='tab:red', linewidth=2, label='Fast')
    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_hist.set_xlim(0.1, 10)
    ax_hist.set_ylim(1e-4, 10)

    ax_hist.set_ylabel(r'PDF', fontsize=14)

    ax_hist.axvline(1.0, color='k', linestyle='dotted', linewidth=1, alpha=0.6, zorder=0)
    ax_hist.grid(True, which='both', linestyle='-', alpha=0.2)
    ax_hist.legend(ncols=3, loc='upper right', fontsize=16)
    
    # ax_hist.hist(ani_valid, bins=ani_bins, color='tab:blue', alpha=0.8, 
    #              edgecolor='black', linewidth=1.0, log=True, density=True)
                 
    ax_hist.set_xlabel(r'Temperature Anisotropy ($T_{\perp} / T_{\parallel}$)', fontsize=16, fontweight='bold')
    ax_hist.set_ylabel('Probability Density', fontsize=16, fontweight='bold')
    ax_hist.set_title('(b) Anisotropy Distributions', fontsize=18, fontweight='bold')
    
    ax_hist.grid(True, which='major', linestyle='-', alpha=0.6, color='gray', linewidth=0.8)
    ax_hist.grid(True, which='minor', linestyle=':', alpha=0.4, color='gray', linewidth=0.6)
    ax_hist.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position("right")
    ax_hist.set_xlim([0.2, 10])

    # --- BOTTOM PANELS: BRAZIL PLOTS ---
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1], sharey=ax_b1)
    ax_b3 = fig.add_subplot(gs[1, 2], sharey=ax_b1)
    cbar_ax_bot = fig.add_subplot(gs[1, 3])

    for ax_b in [ax_b1, ax_b2, ax_b3]:
        ax_b.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax_b.set_xlim([0.01, 20])
        ax_b.set_ylim([0.07, 10])

    plt.setp(ax_b2.get_yticklabels(), visible=False)
    plt.setp(ax_b3.get_yticklabels(), visible=False)
    ax_b1.set_ylabel(r'Temperature Anisotropy ($T_\perp / T_\parallel$)', fontsize=16, fontweight='bold')

    plot_brazil_panel(ax_b1, beta_para_core[flag_all], anisotropy[flag_all], 
                      title='(c) All Solar Wind Events', PRINT_MOMENTS=True, SHOW_LEGEND=True, CMAP='Blues', NBINS=NBINS)
    
    plot_brazil_panel(ax_b2, beta_para_core[flag_slow], anisotropy[flag_slow], 
                      title='(d) Slow Wind', PRINT_MOMENTS=True, SHOW_LEGEND=False, CMAP='Blues', NBINS=NBINS)
    
    mesh = plot_brazil_panel(ax_b3, beta_para_core[flag_fast], anisotropy[flag_fast], 
                      title='(e) Fast Wind', PRINT_MOMENTS=True, SHOW_LEGEND=False, CMAP='Blues', NBINS=NBINS)

    if mesh:
        cbar_bot = fig.colorbar(mesh, cax=cbar_ax_bot)
        cbar_bot.set_label(r'$\mathbf{p(\beta_{\parallel}, \alpha) = n/(N \Delta \beta_{\parallel} \Delta \alpha)}$', fontsize=16, fontweight='bold')

    # Apply final global margins
    plt.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.08)

    plt.savefig('./figures/Composite_Brazil_plot_new.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('./figures/Composite_Brazil_plot_new.png', format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == '__main__':
    file_path = 'selected_dates_updated_v2.pkl'
    file_path2 = 'updated_flags_intervals_v2.pkl'

    with open(file_path2, 'rb') as f:
        updated_flags2 = pickle.load(f)

    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    loaded_list = [date for date_list in loaded_dict.values() for date in date_list]
    cutoff = datetime(2022, 1, 1) 
    loaded_list = [d for d in loaded_list if datetime.strptime(d, "%Y-%m-%d") > cutoff]

    xr_ani_list = []
    for date in loaded_list:
        try:
            updated_flag = updated_flags2[date].rename({'time':'Epoch'})

            year, month, day = date.split('-')
            
            xr_ani = cdflib.xarray.cdf_to_xarray(f'./anisotropy_files/Bayesian_MCMC_5D_NCP/SPC_Anisotropy_{date}_peak_track_T_fit4s_mf38_gflag_v2.cdf')
            xr_ani['new_max_flag'] = updated_flag.total_max_flag
            xr_ani = xr_ani.where(xr_ani != -1e31, np.nan)
            xr_ani_list.append(xr_ani)
        except Exception as e:
            print(f'No data found for {date}: {e}')

    if not xr_ani_list:
        print("No valid CDF files found. Exiting.")
        sys.exit()

    xr_ani_merge = xr.concat(xr_ani_list, dim='Epoch')

    kernel_files = glob.glob('/home/michael/Research/E23_Coronal_Hole/spice_data/spice_data/*.bsp')
    spice.initialize(kernel_files)
    spice.install_frame('IAU_SUN')
    measurement_times = xr_ani_merge.Epoch.data
    parker_trajectory_inertial = spice.get_body('SPP', measurement_times)
    r_sun = 696340
    one_au = 1.496e8
    r_spc = parker_trajectory_inertial.distance.value/r_sun
    r_spc_au = parker_trajectory_inertial.distance.value/one_au

    file_list = glob.glob("./Isopoly_Profiles/v1au=*kms.csv")

    models = []
    for filename in file_list:
        match = re.search(r'v1au=(\d+)kms', filename)
        if match:
            speed = int(match.group(1))
            models.append((speed, filename))

    beta_para_core, _ = calculate_parallel_beta_and_anisotropy(xr_ani_merge, DTYPE='corefit')
    anisotropy = xr_ani_merge['T_ani_avg'].data
    vsw = np.linalg.norm(xr_ani_merge.vp1_fit_SC.data, axis=1)

    bx_angle = xr_ani_merge['B_x_flow_angle'].data
    by_angle = xr_ani_merge['B_y_flow_angle'].data

    valid_x = (np.abs(bx_angle) > 5) & (np.abs(bx_angle) < 85)
    valid_y = (np.abs(by_angle) > 5) & (np.abs(by_angle) < 85)

    angle_flag = valid_x | valid_y

    boundary_slow_func = get_boundary_function("./Isopoly_Profiles/v1au=399kms.csv")
    boundary_fast_func = get_boundary_function("./Isopoly_Profiles/v1au=546kms.csv")

    threshold_slow_speeds = boundary_slow_func(r_spc)
    threshold_fast_speeds = boundary_fast_func(r_spc)

    mu = anisotropy
    delta_minus = xr_ani_merge.T_ani_avg_delta_minus.data
    delta_plus = xr_ani_merge.T_ani_avg_delta_plus.data

    p20_upper = delta_plus/mu
    p20_lower = delta_minus/mu

    
    max_allowable_uncert = 0.2 
    strict_uncert_flag = (p20_upper < max_allowable_uncert) & (p20_lower < max_allowable_uncert) & np.isfinite(p20_upper) & np.isfinite(p20_lower)
    valid_anis = (anisotropy > 0.2) & (anisotropy < 9.0) & angle_flag
    max_flag = (xr_ani_merge['new_max_flag'].values > 0)
    
    plot_composite_classifications(
        r_spc, vsw, beta_para_core, anisotropy, 
        models, threshold_slow_speeds, threshold_fast_speeds,
        max_flag, strict_uncert_flag, valid_anis, NBINS=101
    )


    sys.exit()
    log_bins = np.linspace(-1,1, 51)
    alpha_s = anisotropy[valid_anis & max_flag & (vsw < threshold_slow_speeds)]
    alpha_f = anisotropy[valid_anis & max_flag & (vsw > threshold_fast_speeds)]
    counts_s, log_edges_s = np.histogram(np.log10(alpha_s), bins=log_bins, density=True)
    counts_f, log_edges_f = np.histogram(np.log10(alpha_f), bins=log_bins, density=True)

    lin_edges = 10**log_edges_s

    s_points = np.sum(counts_s)
    f_points = np.sum(counts_f)

    pdf_s = counts_s / (s_points) # * np.diff(log_edges))
    pdf_f = counts_f / (f_points)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.stairs(pdf_s, lin_edges, fill=True, alpha=0.4, color='tab:blue', edgecolor='tab:blue', linewidth=2, label='SPC Anisotropy')
    ax.stairs(pdf_f, lin_edges, fill=False, alpha=0.4, color='tab:red', linewidth=2, label='Chi Anisotropy')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 10)
    ax.set_ylim(1e-4, 10)

    ax.set_xlabel(r'$T_{\perp}/T_{\parallel}$', fontsize=14)
    ax.set_ylabel(r'PDF', fontsize=14)

    ax.axvline(1.0, color='k', linestyle='dotted', linewidth=1, alpha=0.6, zorder=0)
    ax.grid(True, which='both', linestyle='-', alpha=0.2)
    ax.legend(loc='upper right', fontsize=16)

    plt.show()