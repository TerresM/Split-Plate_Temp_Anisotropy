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

def theoretical_ridge_slope(anisotropy, Nz):
    denom = (anisotropy - 1.0)*(1.0 - Nz**2) + 1.0
    if denom == 0: return 0
    m_fit = (Nz * (1.0 - anisotropy)) / denom
    Nx = np.sqrt(1 - Nz**2)
    return m_fit * Nx

def plot_conceptual_schematic():
    # Parameters for a clear visual
    A = 2.3
    Theta = 90.0
    V_bulk = 400.0
    V_th_par = 40.0
    
    # Calculate w_perp for axis scaling
    # A = w_perp^2 / w_par^2  => w_perp = w_par * sqrt(A)
    w_perp = V_th_par * np.sqrt(A)
    
    # Grid
    vz = np.linspace(200, 600, 400)
    vx = np.linspace(-200, 200, 400)
    VZ, VX = np.meshgrid(vz, vx)
    
    # 1. Calculate Dist
    F = get_2d_bi_maxwellian(VZ, VX, V_bulk, V_th_par, A, Theta)
    
    # 2. Geometry Lines
    theta_rad = np.radians(Theta)
    
    # B-Field Line (Slope = tan(theta))
    m_b = np.tan(theta_rad)
    y_b = m_b * (vz - V_bulk)
    
    # Ridge Line
    m_ridge = theoretical_ridge_slope(A, np.cos(theta_rad))
    y_ridge = m_ridge * (vz - V_bulk)
    
    # Ridge Peaks (Discrete)
    vz_peaks = np.linspace(300, 500, 9)
    vx_peaks = m_ridge * (vz_peaks - V_bulk)
    
    # 3. Plot
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    
    # Contours
    levs = np.logspace(-2.75, 0, 8)
    ax.contourf(VZ, VX, F, levels=levs, cmap='Blues', norm=colors.LogNorm(), alpha=0.6)
    ax.contour(VZ, VX, F, levels=levs, colors='k', linewidths=0.5, alpha=0.2)
    
    # --- PRINCIPAL AXES (Grey Lines) ---
    # We define the extent to be approx 2.5 sigma to reach the edge of the visible contours
    sigma_scale = 2.5 
    
    # Major Axis (Perpendicular to B)
    # Direction: theta + 90 deg -> (-sin(theta), cos(theta))
    L_maj = w_perp * sigma_scale
    dx_maj = L_maj * np.sin(theta_rad)  # Projection on Vz (rotated) -> -sin
    dy_maj = L_maj * np.cos(theta_rad)  # Projection on Vx (rotated) -> cos
    
    # Since we are rotating +90 deg from B (which is at theta):
    # x_component = cos(theta+90) = -sin(theta)
    # y_component = sin(theta+90) = cos(theta)
    
    ax.plot([V_bulk - (-L_maj * np.sin(theta_rad)), V_bulk + (-L_maj * np.sin(theta_rad))],
            [0 - (L_maj * np.cos(theta_rad)), 0 + (L_maj * np.cos(theta_rad))],
            color='blue', linewidth=4.0, linestyle='-', zorder=0)

    # Minor Axis (Parallel to B)
    # Direction: theta -> (cos(theta), sin(theta))
    L_min = V_th_par * sigma_scale
    ax.plot([V_bulk - L_min * np.cos(theta_rad), V_bulk + L_min * np.cos(theta_rad)],
            [0 - L_min * np.sin(theta_rad), 0 + L_min * np.sin(theta_rad)],
            color='blue', linewidth=4.0, linestyle='-', zorder=0)
    
    # --- LINES ---
    # B-Field (Black)
    ax.plot(vz, y_b, color='black', linestyle='--', linewidth=2.0, label=r'Magnetic Field ($\hat{b}$)')
    
    # Ridge (Red)
    ax.plot(vz, y_ridge, color='red', linestyle='-', linewidth=3, label=r'VDF Ridge')
    
    # Discrete Points
    ax.plot(vz_peaks, vx_peaks, 'o', color='red', markeredgecolor='k', 
            markersize=6, zorder=10)
    
    # Center Dot
    ax.plot(V_bulk, 0, 'ko', markersize=6)
    
    # Styling
    ax.set_xlabel('$v_z$ (Look Direction)', fontsize=12)
    ax.set_ylabel('$v_x$ (Transverse)', fontsize=12)
    # ax.set_title(r'Geometric Signature of Anisotropy ($T_\perp > T_\parallel$)', fontsize=13)
    ax.set_title(f'2D Model | $\\alpha = {A}$, $\\theta ={Theta}$')
    ax.set_xlim(245, 555)
    ax.set_ylim(-160, 160)
    ax.set_aspect('equal')
    
    # ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.1)

    plt.show()

if __name__ == "__main__":
    plot_conceptual_schematic()