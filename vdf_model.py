import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==========================================
# 1. PHYSICS & GEOMETRY KERNELS
# ==========================================

def get_2d_bi_maxwellian(vz_grid, vx_grid, v_bulk, v_th_par, anisotropy, theta_deg):
    """
    Returns the 2D VDF value on a meshgrid (Vz, Vx).
    B-field is assumed to be in the X-Z plane tilted by theta_deg from Z.
    """
    theta_rad = np.radians(theta_deg)
    Nz = np.cos(theta_rad)
    Nx = np.sin(theta_rad)
    
    # Grid vectors
    # V vector: (vx, 0, vz)
    # B vector: (Nx, 0, Nz)
    
    # Parallel Component: v_par = v . b
    # Shift vz by v_bulk
    vz_shifted = vz_grid - v_bulk
    v_par = vx_grid * Nx + vz_shifted * Nz
    
    # Perpendicular Component magnitude squared: v_perp^2 = v^2 - v_par^2
    v_mag_sq = vx_grid**2 + vz_shifted**2
    v_perp_sq = v_mag_sq - v_par**2
    
    # Thermal speeds
    # A = T_perp / T_par = w_perp^2 / w_par^2
    w_par_sq = v_th_par**2
    w_perp_sq = w_par_sq * anisotropy
    
    # Exponent
    arg = -(v_par**2 / w_par_sq) - (v_perp_sq / w_perp_sq)
    
    # VDF
    f_2d = np.exp(arg)
    
    # Normalize peak to 1 for plotting convenience
    return f_2d / np.max(f_2d)

def theoretical_ridge_slope(anisotropy, Nz):
    """
    Calculates the slope m where vx = m * (vz * Nx).
    Using the derived formula: m = (1-A) / [ (A-1)(1-Nz^2) + 1 ]
    Wait, let's use the explicit form derived earlier:
    m_fit = Nz(1-A) / [ (A-1)(1-Nz^2) + 1 ]
    Note: The x-axis for fitting is (vz * Nx), so vx = m_fit * (vz * Nx).
    The physical slope in vx-vz plane is m_phys = m_fit * Nx.
    """
    # Slope relative to the Fitting Axis (X = Vz*sin(theta))
    denom = (anisotropy - 1.0)*(1.0 - Nz**2) + 1.0
    if denom == 0: return 0
    m_fit = (Nz * (1.0 - anisotropy)) / denom
    
    # Slope relative to Vz (for plotting in Vx vs Vz plane)
    # vx = m_fit * (vz * sin(theta))
    # vx = (m_fit * Nx) * vz
    Nx = np.sqrt(1 - Nz**2)
    m_physical = m_fit * Nx
    return m_physical

def linear_model(x, a, b):
    return a * x + b

# ==========================================
# 2. SIMULATION GENERATOR
# ==========================================
def anisotropy_from_visual_slope(slope_vz, Nz, Nx):
    """
    Calculates Anisotropy (A) given the visual slope m = vx / vz.
    
    Parameters:
    -----------
    slope_vz : float
        The slope from fitting vx vs vz.
    Nz : float
        Z-component of B-field unit vector (cos theta).
    Nx : float
        X-component of B-field unit vector (sin theta).
        
    Returns:
    --------
    A : float
        Anisotropy (T_perp / T_par).
    """
    # 1. The Relationship
    # m_visual = (Nx * Nz * (1-A)) / [ (A-1)*Nx^2 + 1 ]
    
    # 2. Inversion Derivation
    # Let D = (A-1)
    # m = (Nx * Nz * -D) / (D * Nx^2 + 1)
    # m * (D * Nx^2 + 1) = -D * Nx * Nz
    # m * D * Nx^2 + m = -D * Nx * Nz
    # D * (m * Nx^2 + Nx * Nz) = -m
    # D = -m / (Nx * (m * Nx + Nz))
    # A = D + 1
    
    denominator = Nx * (slope_vz * Nx + Nz)
    
    if abs(denominator) < 1e-9:
        return np.nan
        
    A = (-slope_vz / denominator) + 1.0
    
    # Validation
    if np.isscalar(A):
        return A if A > 0 else np.nan
    else:
        A[A <= 0] = np.nan
        return A
    
def generate_simulation_data(A_true=2.3, Theta_true=20.0, V_bulk=400.0, V_th_par=35.0):
    # Instrument model parameters.
    Noise_Floor = 1.0   # pA
    Peak_Current = 1000.0 # pA
    
    # Derived Geometry
    theta_rad = np.radians(Theta_true)
    Nz = np.cos(theta_rad)
    Nx = np.sin(theta_rad)
    
    # --- 1D Reduced VDF Simulation ---
    # Velocity Grid (Instrument)
    vz_grid = np.geomspace(138.0, 850.0, 32)
    # vz_grid = np.geomspace(250, 550, 16)
    
    # Effective Thermal Speed (Width of 1D RDF)
    T_eff_norm = Nz**2 + A_true * Nx**2
    v_th_eff = V_th_par * np.sqrt(T_eff_norm)
    
    # Define the ideal reduced distribution function
    current = np.exp(-1.0 * ((vz_grid - V_bulk)/v_th_eff)**2)

    current_ideal = Peak_Current * current
    current_noise = np.random.normal(0, Noise_Floor, size=len(vz_grid))

    current_model = (current_ideal + current_noise) + 2.0

    # Convert to RDF
    rdf_model = current_model / (vz_grid**2)
    rdf_model = (rdf_model)/np.max(rdf_model)

    rdf_ideal = (current_ideal + 2.0) / (vz_grid**2)
    rdf_ideal = rdf_ideal/np.max(rdf_ideal)
    
    # Ideal Vx (The Ridge)
    m_phys = theoretical_ridge_slope(A_true, Nz)
    vx_ideal = m_phys * vz_grid  # Note: assuming V_bulk shift cancels in slope logic or applies to full V
    # Actually, the ridge is centered at V_bulk? 
    # The derivation assumed V centered at 0. For shifted Maxwellian:
    # vx_ridge = m * (vz) is only true if u_x = m * u_z.
    # If flow is purely radial (u_x=0, u_z=V_bulk), the ridge is offset.
    # However, standard analysis assumes we fit V_perp moments relative to the beam center.
    # Let's assume the "Measured Vx" includes the bulk flow offset if any.
    # For a radial beam, mean Vx is 0. The ridge is the correlation of thermal fluctuations.
    # Let's model the Vx relative to the center of the beam (V_bulk).
    
    # Vz relative to bulk
    dvz = vz_grid - V_bulk
    vx_ideal = m_phys * dvz # The ridge passes through (V_bulk, 0)
    
    # Vx Error
    # Sigma_vx approx Vz * (Noise / Signal)
    # We use the full Vz for the lever arm effect
    # sigma_vx = 0.1*(sigma_I / (I_ideal + 1e-3))
    sigma_vx = 10.0 * (current_noise / (current_ideal + 0.01)) 
    
    vx_noise = np.random.normal(0, 1, len(vz_grid)) * sigma_vx
    vx_model = vx_ideal + vx_noise
    
    return {
        'vz': vz_grid,
        'vx_meas': vx_model,
        'vx_err': sigma_vx,
        'rdf_meas': rdf_model,
        'rdf_ideal': rdf_ideal,
        # 'rdf_err': sigma_rdf,
        'params': (A_true, Theta_true, V_bulk, V_th_par)
    }

# ==========================================
# 3. PLOTTING ROUTINE
# ==========================================
def plot_ridge_demonstration(A_true=2.3, Theta_true=20.0, V_bulk=400.0, V_th_par=55.0):
    THRES = 0.35
    
    # --- 1. Run Sim & Fit ---
    data = generate_simulation_data(A_true, Theta_true, V_bulk, V_th_par)
    vz = data['vz']
    vx_meas = data['vx_meas']
    rdf_meas = data['rdf_meas']
    A, Theta, V_bulk, V_th_par = data['params']
    
    theta_rad = np.radians(Theta)
    Nz = np.cos(theta_rad)
    Nx = np.sin(theta_rad)
    m_bfield = np.tan(theta_rad)

    weights = rdf_meas / np.max(rdf_meas)
    mask = weights > THRES 
    popt, _ = curve_fit(linear_model, vz[mask], vx_meas[mask], 
                        sigma=1/weights[mask], method='lm')
    m_fitted = popt[0]
    b_fitted = popt[1]
    T_ani = anisotropy_from_visual_slope(m_fitted, Nz, Nx)
    
    # --- 2. Prepare Grids ---
    vz_fine = np.linspace(138, 800, 200)
    vx_fine = np.linspace(-500, 500, 200)
    VZ_2D, VX_2D = np.meshgrid(vz_fine, vx_fine)
    y_bfield = m_bfield * (vz_fine - V_bulk)
    F_2D = get_2d_bi_maxwellian(VZ_2D, VX_2D, V_bulk, V_th_par, A, Theta)
    
    # --- 3. Setup Figure (Manual Layout) ---
    # We disable constrained_layout to have full manual control
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8.5), height_ratios=[1, 2])
    
    # Manually adjust margins to ensure space for the bottom colorbar
    # bottom=0.12 gives roughly 12% of figure height for the footer
    plt.subplots_adjust(left=0.15, right=0.9, top=0.92, bottom=0.12, hspace=0.1)

    # --- PANEL 1 ---
    ax1.plot(vz, rdf_meas, color='k')
    # ax1.plot(vz, data['rdf_ideal'], color='tab:blue', linestyle='--', linewidth=2, label='Ideal 1D Model')
    ax1.set_ylabel('Reduced VDF\n[Log scale]')
    ax1.set_yscale('log')
    ax1.axhline(THRES, linestyle='--', color='red', linewidth=2)
    # ax1.set_ylim(5e-3, 1.5)
    ax1.legend(loc='upper right', frameon=False, fontsize='small')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Simulated Measurement ($A={A}$, $\\theta={Theta}^\circ$, Noise=1pA)\n Measured $A={T_ani:.2f}$', fontsize=12)
    
    # --- PANEL 2 ---
    levs = np.logspace(-2, 0, 10)
    ax2.contourf(VZ_2D, VX_2D, F_2D, levels=levs, cmap='Blues', norm=colors.LogNorm())
    
    ax2.scatter(vz, vx_meas, color='grey', alpha=0.5, s=20)
    sc = ax2.scatter(vz[mask], vx_meas[mask], c=weights[mask], cmap='viridis', 
                     edgecolor='k', s=50, zorder=10, label='Measured $v_x$')
    
    m_true = theoretical_ridge_slope(A, np.cos(np.radians(Theta)))
    y_true = m_true * (vz_fine - V_bulk)
    ax2.plot(vz_fine, y_true, 'r--', linewidth=2, label='Theoretical Ridge')
    ax2.plot(vz_fine, y_bfield, color='green', label='B-Field')
    ax2.plot(vz_fine, m_fitted * vz_fine + b_fitted, 'k-', linewidth=2, label='Fitted Ridge')
    
    ax2.set_xlabel('$v_z$ [km/s]')
    ax2.set_ylabel('$v_x$ [km/s]')
    ax2.set_ylim(-350, 350) # Set this before aspect ratio
    ax2.legend(loc='upper left', framealpha=0.9, fontsize='small')
    ax2.grid(True, alpha=0.3)

    # --- 4. Colorbar Attachment ---
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("bottom", size="5%", pad=0.6)
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label('Measurement Weight (Normalized RDF)')

    # --- 5. Aspect Ratio & Alignment (THE FIX) ---
    # A. Force Equal Aspect Ratio
    ax2.set_xlim([200,800])
    ax2.set_aspect('equal', adjustable='box')
    
    # B. Force a draw so Matplotlib calculates the final shrunken size of ax2
    fig.canvas.draw()
    
    # C. Get the positions
    box2 = ax2.get_position()
    box1 = ax1.get_position()
    
    # D. Manually force ax1 to match ax2's width and horizontal start
    ax1.set_position([box2.x0, box1.y0, box2.width, box1.height])
    
    # E. Sync X-Limits after the resize
    ax1.set_xlim(ax2.get_xlim())
    ax1.set_xticklabels([])

    plt.show()

if __name__ == "__main__":
    plot_ridge_demonstration(A_true=2.3, Theta_true=20)