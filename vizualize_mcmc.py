import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

# --- 1. Helpers & Physics Engine ---

def calc_anisotropy_from_slope_normalized(m, theta_rad):
    """
    Inverts the slope equation to find anisotropy alpha = T_perp / T_par.
    Assumes fit: Vx = m * (Vz * sin(theta))
    """
    s = np.sin(theta_rad)
    c = np.cos(theta_rad)
    
    # Formula: A = (cos - m*cos^2) / (cos + m*sin^2)
    numerator = c - (m * c**2)
    denominator = c + (m * s**2)
    
    if abs(denominator) < 1e-6: return np.nan
    
    alpha = numerator / denominator
    return alpha

def generate_numerical_event_for_viz(alpha, theta_deg, n_steps=25, noise_amp=0.05):
    theta = np.radians(theta_deg)
    Nz = np.cos(theta)
    Nx = np.sin(theta)

    v_th_par = 20.0 
    v_th_perp = np.sqrt(alpha) * v_th_par
    
    # Define Sweep Range based on Physics
    T_eff_z = (v_th_par**2 * Nz**2) + (v_th_perp**2 * Nx**2)
    sweep_range = 2.5 * np.sqrt(T_eff_z)
    
    vz_array = np.linspace(-sweep_range, sweep_range, n_steps)
    vx_grid = np.linspace(-sweep_range*1.5, sweep_range*1.5, 200)
    
    # 2D PDF Construction
    VZ, VX = np.meshgrid(vz_array, vx_grid, indexing='ij') 
    v_par = VZ * Nz + VX * Nx
    v_perp = -VZ * Nx + VX * Nz
    exponent = -((v_par + 40)**2 / v_th_par**2) - (v_perp**2 / v_th_perp**2)
    
    pdf_ideal = np.exp(exponent)
    pdf_ideal /= np.max(pdf_ideal)
    
    # Add Noise
    noise = np.random.normal(0, noise_amp, pdf_ideal.shape)
    pdf_noisy = np.maximum(pdf_ideal + noise, 0)
    
    # Moments Integration
    flux_profile = np.trapz(pdf_noisy, vx_grid, axis=1)
    
    # Normalize Flux
    if np.max(flux_profile) > 0:
        flux_norm = flux_profile / np.max(flux_profile)
    else:
        flux_norm = flux_profile * 0
    
    numerator = np.trapz(VX * pdf_noisy, vx_grid, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        vx_measured = numerator / flux_profile
    
    # --- Define Fit Coordinates (Normalized Method) ---
    # X_fit = Vz * sin(theta)
    x_fit_coords = vz_array * Nx
    
    # Calculate Theoretical Slope m for these coords
    # m = [cos*(1-A)] / [cos^2 + A*sin^2]
    num = Nz * (1 - alpha)
    den = (Nz**2) + alpha * (Nx**2)
    m_true = num / den if abs(den) > 1e-8 else 0

    return x_fit_coords, vz_array, vx_measured, flux_norm, m_true, pdf_ideal, sweep_range * 1.5

# --- 2. Visualization Function ---

def visualize_complete_fit(alpha=2.0, theta=45.0, noise_amp=0.03, threshold=0.3):
    
    # Generate Data
    res = generate_numerical_event_for_viz(alpha, theta, n_steps=16, noise_amp=noise_amp)
    x_fit_coords, vz_raw, vx_sim, flux_sim, m_true, pdf_2d, grid_lim = res
    
    # Fit Logic (Apply Threshold)
    mask = flux_sim > threshold
    
    def model(x, m, c): return m*x + c
    
    # Perform Fit on Normalized Coords (Vz * sin(theta))
    if np.sum(mask) > 3:
        popt, _ = curve_fit(model, x_fit_coords[mask], vx_sim[mask], sigma=1.0/flux_sim[mask])
        m_fit = popt[0]
        c_fit = popt[1]
    else:
        m_fit = np.nan
        c_fit = np.nan

    # Calculate Anisotropy from Fit
    theta_rad = np.radians(theta)
    Nz = np.cos(theta_rad)
    Nx = np.sin(theta_rad)
    
    T_ani_measured = calc_anisotropy_from_slope_normalized(m_fit, theta_rad)
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], 
                           wspace=0.1, hspace=0.1)

    # --- Main Panel: 2D Contours, Scatter ---
    ax_main = fig.add_subplot(gs[1, 0])
    
    # A. Background Contours (Physics)
    v_fine = np.linspace(-grid_lim, grid_lim, 200)
    VZ, VX = np.meshgrid(v_fine, v_fine) # Raw Vz, Vx
    
    # Transform Vz to Plot X-axis
    X_PLOT = VZ * Nx 
    
    vp = VZ*Nz + VX*Nx
    vpp = -VZ*Nx + VX*Nz
    v_tpar, v_tperp = 20.0, 20.0 * np.sqrt(alpha)
    PDF = np.exp(-((vp + 40)**2/v_tpar**2) - (vpp**2/v_tperp**2))
    
    ax_main.contourf(X_PLOT, VX, PDF, levels=15, cmap='Greys', alpha=0.3)
    
    # B. Scatter Points (Using Normalized X Coords)
    ax_main.scatter(x_fit_coords[~mask], vx_sim[~mask], facecolors='none', edgecolors='gray', 
                    s=40, alpha=0.4, label='Rejected (< Threshold)')
    ax_main.scatter(x_fit_coords[mask], vx_sim[mask], c=flux_sim[mask], cmap='viridis', 
                    edgecolors='k', s=80, zorder=10, label='Included Data')
    
    # C. Slopes
    x_range_plot = np.array([-grid_lim*Nx, grid_lim*Nx])
    
    # 1. TRUE SLOPE (Blue Dotted)
    ax_main.plot(x_range_plot, m_true * x_range_plot, color='blue', linestyle=':', linewidth=3, 
                 label=f'True Slope (m={m_true:.2f})')
    
    # 2. FITTED SLOPE (Red Solid)
    if np.isfinite(m_fit):
        # We now plot over the full range (same as x_range_plot)
        ax_main.plot(x_range_plot, m_fit * x_range_plot + c_fit, 'r-', linewidth=3, 
                     alpha=0.8, label=f'Fit (m={m_fit:.2f})')
        
    # D. Magnetic Field Vector
    # The B-field in Raw Space is (Vz, Vx) = (cos theta, sin theta)
    # In Plot Space:
    #   X_arrow = Vz * Nx = cos(theta) * sin(theta)
    #   Y_arrow = Vx      = sin(theta)
    arrow_len = grid_lim * 0.7
    arr_x = arrow_len * Nz * Nx
    arr_y = arrow_len * Nx
    
    ax_main.arrow(0, 0, arr_x, arr_y, color='red', width=grid_lim*0.015, 
                  head_width=grid_lim*0.06, zorder=15, label=r'$\mathbf{B}$ Field')

    # Formatting Main Panel
    ax_main.set_xlim(-grid_lim*abs(Nx), grid_lim*abs(Nx))
    ax_main.set_ylim(-grid_lim, grid_lim)
    ax_main.set_xlabel(r'$v_z \cdot \sin(\theta)$ (Projected Radial) [km/s]', fontsize=12)
    ax_main.set_ylabel(r'$v_x$ (Transverse) [km/s]', fontsize=12)
    ax_main.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    
    # Add Results Text Box
    res_str = '\n'.join((
        r'$\mathbf{Simulation\ Results}$',
        r'$A_{true} = %.2f$' % (alpha, ),
        r'$A_{meas} = %.2f$' % (T_ani_measured, ),
        r'Err = $%.1f$%%' % (abs(T_ani_measured - alpha)/alpha * 100, )
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax_main.text(0.95, 0.05, res_str, transform=ax_main.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # --- Top Panel: Flux Profile (vs Normalized X) ---
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.plot(x_fit_coords, flux_sim, 'k-o', alpha=0.3)
    ax_top.plot(x_fit_coords[mask], flux_sim[mask], 'o', color='orange')
    ax_top.axhline(threshold, color='red', linestyle='--', label='Threshold')
    
    title = f'SPC Normalized Fit: $T_\perp/T_\parallel={alpha}$, $\\theta={theta}^\circ$'
    ax_top.set_title(title, fontsize=14)
    ax_top.set_ylabel('Norm. Flux')
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # --- Right Panel: Marginal Vx ---
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    vx_marg = np.sum(pdf_2d, axis=0) 
    v_fine_y = np.linspace(-grid_lim, grid_lim, len(vx_marg))
    
    ax_right.plot(vx_marg, v_fine_y, 'k-', alpha=0.5)
    ax_right.fill_betweenx(v_fine_y, 0, vx_marg, color='gray', alpha=0.2)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_xlabel(r'$f(v_x)$')
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_complete_fit(alpha=2.5, theta=55.0, noise_amp=0.02, threshold=0.1)