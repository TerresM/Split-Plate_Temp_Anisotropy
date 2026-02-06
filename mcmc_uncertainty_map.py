import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from matplotlib.colors import LogNorm

# ==========================================
# 1. PHYSICS & ANALYSIS FUNCTIONS
# ==========================================

def linear_model(x, a, b):
    return a * x + b

def calc_anisotropy_from_slope_normalized(m, theta_rad):
    """
    Inverts the slope 'm' to find Anisotropy alpha = T_perp/T_par.
    Assumes fit: Vx = m * (Vz * sin(theta))
    """
    s = np.sin(theta_rad)
    c = np.cos(theta_rad)
    
    # The algebraic inversion we derived:
    # A = (cos - m*cos^2) / (cos + m*sin^2)
    numerator = c - (m * c**2)
    denominator = c + (m * s**2)
    
    if abs(denominator) < 1e-6: 
        return np.nan
        
    return numerator / denominator

def find_slopes_sim(xdata, ydata, weights):
    """
    Fits the linear trend of the ridge.
    Weights are Flux-based (higher flux = more importance).
    """
    try:
        # Filter for valid data
        mask = (weights > 1e-4) & np.isfinite(ydata) & np.isfinite(xdata)
        
        if np.sum(mask) < 4: 
            return [np.nan, np.nan], None
        
        # Weighted Least Squares
        # Sigma is inversely proportional to weights (Flux)
        popt, pcov = curve_fit(linear_model, xdata[mask], ydata[mask], 
                               sigma=1.0/weights[mask], absolute_sigma=False)
        return popt, pcov
    except:
        return [np.nan, np.nan], None

# ==========================================
# 2. NUMERICAL INTEGRATION ENGINE (No N_Particles)
# ==========================================

def generate_numerical_event(alpha, theta_deg, n_steps=20, noise_amp=0.05):
    """
    Simulates the instrument response by numerically integrating the 2D VDF.
    
    Parameters:
    - alpha:      True Anisotropy (T_perp / T_par)
    - theta_deg:  Magnetic Field Angle
    - n_steps:    Number of Vz energy windows
    - noise_amp:  Amplitude of noise added to the PDF (relative to Peak=1)
    """
    
    # A. Geometry
    theta = np.radians(theta_deg)
    Nz = np.cos(theta)
    Nx = np.sin(theta)
    
    # B. Define Thermal Widths (Arbitrary units, T_par=100)
    v_th_par = 20.0 
    v_th_perp = np.sqrt(alpha) * v_th_par
    
    # Calculate sweep range based on effective thermal width along Z
    T_eff_z = (v_th_par**2 * Nz**2) + (v_th_perp**2 * Nx**2)
    sweep_range = 3.0 * np.sqrt(T_eff_z) # 3 sigma sweep
    
    # C. Define Grids
    # vz_array: The energy steps the instrument scans
    vz_array = np.linspace(-sweep_range, sweep_range, n_steps)
    
    # vx_grid: The fine grid used for integration (Transverse)
    vx_grid = np.linspace(-sweep_range*2, sweep_range*2, 100)
    
    # Meshgrid for 2D calculation (Vz, Vx)
    VZ, VX = np.meshgrid(vz_array, vx_grid, indexing='ij') 
    
    # D. Calculate Ideal 2D VDF (Bi-Maxwellian)
    #    v_par = vz*cos + vx*sin
    #    v_perp_in_plane = -vz*sin + vx*cos
    v_par = VZ * Nz + VX * Nx
    v_perp = -VZ * Nx + VX * Nz
    
    exponent = -(v_par**2 / v_th_par**2) - (v_perp**2 / v_th_perp**2)
    pdf_ideal = np.exp(exponent)
    
    # Normalize PDF peak to 1.0
    pdf_ideal /= np.max(pdf_ideal)
    
    # E. Add Noise (Simulating Electronics/fitting noise)
    #    We add noise to the curve, not particles
    noise = np.random.normal(0, noise_amp, pdf_ideal.shape)
    pdf_noisy = pdf_ideal + noise
    
    # Clip negatives (physics check)
    pdf_noisy = np.maximum(pdf_noisy, 0)
    
    # F. Integrate to get Moments (The "Ridge")
    
    # 1. Flux (Zeroth Moment) -> The weights for our fit
    flux_profile = np.trapz(pdf_noisy, vx_grid, axis=1)
    # Normalize Flux to 1
    if np.max(flux_profile) > 0:
        flux_norm = flux_profile / np.max(flux_profile)
    else:
        flux_norm = flux_profile * 0

    # 2. Transverse Velocity (Measured Vx) -> The Y data for our fit
    #    Mean Vx = Integral(vx * f) / Integral(f)
    numerator = np.trapz(VX * pdf_noisy, vx_grid, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vx_measured = numerator / flux_profile
        
    # G. Prepare Fit Data (Normalized Slope Method)
    #    X_data = Vz * sin(theta)
    X_data = vz_array * Nx 
    Y_data = vx_measured   
    
    return X_data, Y_data, flux_norm, theta

# ==========================================
# 3. SIMULATION LOOP (With Map Generation)
# ==========================================

def run_simulation_numerical():
    # Grid Setup
    angles = np.linspace(-85, 85, 30)
    # Filter out near-zero angles where Nx=0 (Singularity)
    angles = angles[np.abs(angles) > 2]
    
    anisotropies = np.logspace(np.log10(0.1), np.log10(10), 30)
    
    map_delta_plus = np.zeros((len(angles), len(anisotropies)))
    map_delta_minus = np.zeros((len(angles), len(anisotropies)))
    
    print("Running Numerical Integration Simulation...")
    print(f"Grid: {len(angles)} Angles x {len(anisotropies)} Anisotropies")
    
    # Iterate over parameter space
    for i, ang in enumerate(tqdm(angles)):
        for j, ani_true in enumerate(anisotropies):
            
            diffs = []
            
            # Run 50 Trials per point to build uncertainty statistics
            # In numerical method, 'noise_amp' changes the result each time
            for _ in range(50):
                X, Y, W, theta_rad = generate_numerical_event(ani_true, ang, noise_amp=0.03)
                
                # Fit the line
                fit_vals, _ = find_slopes_sim(X, Y, W)
                slope = fit_vals[0]
                
                if np.isnan(slope): continue

                # Invert to get Anisotropy
                ani_est = calc_anisotropy_from_slope_normalized(slope, theta_rad)
                
                if np.isfinite(ani_est) and ani_est > 0:
                    diffs.append(ani_est - ani_true)
            
            # Calculate Statistics (Percentiles) for Deltas
            diffs = np.array(diffs)
            
            if len(diffs) > 10:
                # 84th percentile corresponds to +1 Sigma
                # 16th percentile corresponds to -1 Sigma
                p16 = np.percentile(diffs, 16)
                p84 = np.percentile(diffs, 84)
                
                # Delta Plus is distance from Truth to Upper Bound
                # Delta Minus is distance from Truth to Lower Bound
                # Since 'diffs' = Est - True, we just take absolute values
                # Note: This definition assumes the error distribution is roughly centered
                map_delta_plus[i, j]  = np.abs(p84)
                map_delta_minus[i, j] = np.abs(p16)
            else:
                map_delta_plus[i, j] = np.nan
                map_delta_minus[i, j] = np.nan

    return angles, anisotropies, map_delta_plus, map_delta_minus

# ==========================================
# 4. EXECUTION AND VISUALIZATION
# ==========================================

if __name__ == "__main__":
    angles, anis, d_plus, d_minus = run_simulation_numerical()

    # Create Maps
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # We transpose maps for pcolormesh (pcolormesh expects [y, x])
    # X axis = Anisotropy, Y axis = Angle
    # The loops were (Angle, Anisotropy), so shape is (Ang, Ani)
    # This matches pcolormesh logic if X=Anis, Y=Angles
    
    X, Y = np.meshgrid(anis, angles)
    
    # Delta Plus Map 
    # We normalize by the True Anisotropy to get Relative Uncertainty
    c1 = ax[0].pcolormesh(X, Y, d_plus, cmap='inferno', shading='auto', norm=LogNorm())
    ax[0].set_xscale('log')
    ax[0].set_title(r'$\Delta_+$ Relative Uncertainty (Numerical)')
    ax[0].set_ylabel('Incident Angle (deg)')
    ax[0].set_xlabel('True Anisotropy')
    plt.colorbar(c1, ax=ax[0], label=r'$\Delta_+ / A_{true}$')
    
    # Delta Minus Map 
    c2 = ax[1].pcolormesh(X, Y, d_minus, cmap='inferno', shading='auto', norm=LogNorm())
    ax[1].set_xscale('log')
    ax[1].set_title(r'$\Delta_-$ Relative Uncertainty (Numerical)')
    ax[1].set_xlabel('True Anisotropy')
    ax[1].set_yticks([]) # Share Y axis visually
    plt.colorbar(c2, ax=ax[1], label=r'$\Delta_- / A_{true}$')
    
    plt.suptitle('SPC Anisotropy Uncertainty Maps (Numerical Integration Method)', fontsize=14)
    plt.tight_layout()
    plt.show()