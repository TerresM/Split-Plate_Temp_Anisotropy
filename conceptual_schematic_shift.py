import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def get_2d_bi_maxwellian(vz_grid, vx_grid, v_bulk, v_th_par, anisotropy, theta_deg):
    theta_rad = np.radians(theta_deg)
    Nz = np.cos(theta_rad)
    Nx = np.sin(theta_rad)
    
    vz_shifted = vz_grid - v_bulk
    v_par = vx_grid * Nx + vz_shifted * Nz
    v_mag_sq = vx_grid**2 + vz_shifted**2
    v_perp_sq = v_mag_sq - v_par**2
    
    w_par_sq = v_th_par**2
    w_perp_sq = w_par_sq * anisotropy
    
    arg = -(v_par**2 / w_par_sq) - (v_perp_sq / w_perp_sq)
    return np.exp(arg)

def theoretical_ridge_slope_vz(anisotropy, Nz):
    # Slope m where vx = m * (vz - uz)
    denom = (anisotropy - 1.0)*(1.0 - Nz**2) + 1.0
    if denom == 0: return 0
    m_fit = (Nz * (1.0 - anisotropy)) / denom
    Nx = np.sqrt(1 - Nz**2)
    return m_fit * Nx

def plot_conceptual_schematic():
    # Parameters
    A = 4.0
    Theta = 30.0 # Must be > 0 for bx scaling to make sense
    V_bulk = 400.0
    V_th_par = 40.0
    
    # 1. Physics Calcs
    theta_rad = np.radians(Theta)
    Nx = np.sin(theta_rad)
    Nz = np.cos(theta_rad)
    
    # Define Grid in Vz-Vx space first (Physical Space)
    vz_phys = np.linspace(200, 600, 400)
    vx_phys = np.linspace(-200, 200, 400)
    VZ_phys, VX_phys = np.meshgrid(vz_phys, vx_phys)
    
    F = get_2d_bi_maxwellian(VZ_phys, VX_phys, V_bulk, V_th_par, A, Theta)

    # 2. Coordinate Transformation for Plotting
    # X_plot = (vz - u_z) * bx
    # Y_plot = vx - ux (ux=0 here)
    
    X_grid_plot = (VZ_phys - V_bulk) * Nx
    Y_grid_plot = VX_phys 
    
    # 3. Line Definitions in Transformed Space
    # Original: vx = m * (vz - u_z)
    # New:      Y = m_new * X   =>   vx = m_new * (vz - u_z) * Nx
    # Therefore: m_new = m_original / Nx
    
    # A. Ridge Line
    m_ridge_phys = theoretical_ridge_slope_vz(A, Nz)
    m_ridge_plot = m_ridge_phys / Nx
    
    x_line = np.linspace(-100, 100, 200) # Range in (vz-uz)*bx space
    y_ridge = m_ridge_plot * x_line
    
    # B. Magnetic Field Line
    # Phys: vx = tan(theta) * (vz - uz)
    # Plot: m_b_plot = tan(theta) / sin(theta) = 1 / cos(theta)
    m_b_plot = 1.0 / Nz
    y_b = m_b_plot * x_line
    
    # C. Principal Axes
    w_perp = V_th_par * np.sqrt(A)
    scale = 2.5
    
    # Major Axis (Phys slope: -cot(theta))
    # Plot slope: -cot(theta) / sin(theta) = -cos/sin^2
    m_maj_plot = -Nz / (Nx**2)
    
    # For the segment lengths, we project the physical vector onto the new X-axis
    # Phys Vector: L * [-sin, cos] (relative to origin)
    # X_proj = (-L sin(theta)) * sin(theta) = -L sin^2
    # Y_proj = L cos(theta)
    L_maj = w_perp * scale
    dx_maj_plot = -L_maj * (Nx**2) 
    dy_maj_plot = L_maj * Nz
    
    # Minor Axis (Phys slope: tan(theta))
    # Plot slope: 1/cos(theta) (Same as B-field)
    L_min = V_th_par * scale
    dx_min_plot = L_min * Nz * Nx  # (L cos) * sin
    dy_min_plot = L_min * Nx       # L sin
    
    # D. Discrete Points (Ridge Peaks)
    # Phys Vz peaks from previous code
    dvz_peaks = np.linspace(-140, 140, 9) # centered on bulk
    x_peaks_plot = dvz_peaks * Nx
    y_peaks_plot = m_ridge_plot * x_peaks_plot

    # 4. Plot
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    
    # Use contourf on the TRANSFORMED grid
    # Note: X_grid_plot is not uniform anymore, but contourf handles it if inputs are 2D arrays
    levs = np.logspace(-3, 0, 8)
    ax.contourf(X_grid_plot, Y_grid_plot, F, levels=levs, cmap='Blues', norm=colors.LogNorm(), alpha=0.6)
    ax.contour(X_grid_plot, Y_grid_plot, F, levels=levs, colors='k', linewidths=0.5, alpha=0.2)
    
    # Plot Lines
    # Principal Axes
    ax.plot([-dx_maj_plot, dx_maj_plot], [-dy_maj_plot, dy_maj_plot], 
            color='gray', linewidth=1.5, linestyle='-', zorder=4, label='Principal Axes')
    ax.plot([-dx_min_plot, dx_min_plot], [-dy_min_plot, dy_min_plot], 
            color='gray', linewidth=1.5, linestyle='-', zorder=4)

    # B-Field
    ax.plot(x_line, y_b, color='black', linestyle='--', linewidth=2.0, label=r'Magnetic Field ($\hat{b}$)')
    
    # Ridge
    ax.plot(x_line, y_ridge, color='red', linestyle='-', linewidth=2, label=r'Locus of $v_x$ Maxima')
    ax.plot(x_peaks_plot, y_peaks_plot, 'o', color='red', markeredgecolor='k', markersize=6, zorder=10)
    
    # Center
    ax.plot(0, 0, 'ko', markersize=6)
    
    # Styling
    ax.set_xlabel(r'$(v_z - u_z) \hat{b}_x$ [km/s]', fontsize=12)
    ax.set_ylabel(r'$v_x - u_x$ [km/s]', fontsize=12)
    ax.set_title(r'Transformed Space ($v_x$ vs $v_z \sin\theta$)', fontsize=13)
    
    ax.set_xlim(-80, 80)
    ax.set_ylim(-150, 150)
    
    # Note: Aspect equal might look distorted in this transformed space depending on Theta
    # But often useful to see the skew
    # ax.set_aspect('equal') 
    
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.1)

    plt.show()

if __name__ == "__main__":
    plot_conceptual_schematic()