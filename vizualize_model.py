import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit

from matplotlib.colors import LogNorm

plt.ion()

def generate_numerical_event(alpha, theta_deg, n_steps=25, noise_amp=0.03):
    """
    Simulates the instrument response by numerically integrating the 2D VDF.
    """
    # A. Geometry
    theta = np.radians(theta_deg)
    Nz = np.cos(theta)
    Nx = np.sin(theta)
    
    # B. Define Thermal Widths (Arbitrary units, T_par=100)
    v_th_par = 100.0 
    v_th_perp = np.sqrt(alpha) * v_th_par
    
    # C. Determine Sweep Range (3-sigma of the effective width along Vz)
    T_eff_z = (v_th_par**2 * Nz**2) + (v_th_perp**2 * Nx**2)
    sweep_range = 3.0 * np.sqrt(T_eff_z) 
    
    # D. Define Grids
    # vz_array = np.linspace(-sweep_range, sweep_range, n_steps)
    # vx_grid = np.linspace(-sweep_range*1.5, sweep_range*1.5, 100)

    vz_array = np.linspace(200, 900, n_steps)
    vx_grid  = np.linspace(-500, 500, n_steps)

    VZ, VX = np.meshgrid(vz_array, vx_grid, indexing='ij') 

    UZ = 450
    UX =  50
    
    # E. Calculate Ideal 2D VDF (Bi-Maxwellian)
    v_par = (VZ - UZ) * Nz + (VX - UX) * Nx
    v_perp = -(VZ - UZ) * Nx + (VX - UX) * Nz
    exponent = -((v_par)**2 / v_th_par**2) - (v_perp**2 / v_th_perp**2)
    pdf_ideal = np.exp(exponent)
    pdf_ideal /= np.max(pdf_ideal)
    
    # F. Add Noise (Simulating Electronics/fitting noise)
    noise = np.random.normal(0, noise_amp, pdf_ideal.shape)
    pdf_noisy = np.maximum(pdf_ideal + noise, 0)
    
    # G. Integrate Moments (The "Ridge")
    flux_profile = np.trapz(pdf_noisy, vx_grid, axis=1)
    
    if np.max(flux_profile) > 0:
        flux_norm = flux_profile / np.max(flux_profile)
    else:
        flux_norm = flux_profile * 0

    numerator = np.trapz(VX * pdf_noisy, vx_grid, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vx_measured = numerator / flux_profile
        
    # H. Prepare Fit Data (Normalized Slope Method)
    X_data = vz_array * Nx  # Vz * sin(theta)
    Y_data = vx_measured    # Vx
    
    return X_data, Y_data, flux_norm, theta, VX, VZ, pdf_noisy

def linear_model(x, a, b):
    return a * x + b

def calc_anisotropy_from_slope_normalized(m, theta_rad):
    """
    Inverts the slope 'm' to find Anisotropy alpha.
    """
    s = np.sin(theta_rad)
    c = np.cos(theta_rad)
    
    # Formula: A = (cos - m*cos^2) / (cos + m*sin^2)
    numerator = c - (m * c**2)
    denominator = c + (m * s**2)
    
    if abs(denominator) < 1e-6: return np.nan
    return numerator / denominator

def find_slopes_sim(xdata, ydata, weights):
    """
    Fits the linear trend.
    CRITICAL: We must center the X-data (Vz) around the peak before fitting,
    because the slope relationship holds for deviations (delta_Vz).
    """
    # Filter valid data
    mask = (weights > 1e-4) & np.isfinite(ydata) & np.isfinite(xdata)
    
    if np.sum(mask) < 4: 
        return [np.nan, np.nan], None
    
    # Center the Vz data (Shift so the weighted mean is 0)
    # This simulates analyzing fluctuations relative to the bulk speed
    x_valid = xdata[mask]
    w_valid = weights[mask]
    x_center = np.average(x_valid, weights=w_valid)
    x_centered = x_valid - x_center
    
    try:
        popt, pcov = curve_fit(linear_model, x_centered, ydata[mask], 
                               sigma=1.0/w_valid, absolute_sigma=False)
        return popt, pcov
    except:
        return [np.nan, np.nan], None

def generate_numerical_event_realistic(alpha, theta_deg, n_steps = 32,
                                       noise_amp=0.03, 
                                       uz=450.0, ux=0): # <-- New Parameter
    """
    Simulates the instrument with FIXED Velocity limits.
    """
    # A. Geometry
    theta = np.radians(theta_deg)
    Nz = np.cos(theta)
    Nx = np.sin(theta)
    
    # B. Physics: Thermal Widths
    v_th_par = 75.0 
    v_th_perp = np.sqrt(alpha) * v_th_par
    
    # C. HARD LIMITS (Instrument Constraints)
    # Vz measures 200 to 800 km/s
    vz_min, vz_max = 200.0, 800.0
    # Vx measures -400 to 400 km/s
    vx_lim = 400.0
    
    # D. Define Fixed Grids
    # We use e.g., 64 steps for Vz (typical instrument resolution)
    vz_array = np.linspace(vz_min, vz_max, n_steps) 
    # Fine grid for integration within limits
    vx_grid = np.linspace(-vx_lim, vx_lim, n_steps)
    
    VZ, VX = np.meshgrid(vz_array, vx_grid, indexing='ij') 
    
    # E. 2D VDF (Shifted to V_bulk frame)
    # The plasma is moving at V_bulk along Z.
    # Parallel/Perp components relative to the B-field:
    # v_vec = (vx, vz - v_bulk)
    # v_par = (vz - v_bulk)*cos + vx*sin
    
    delta_vz = VZ - uz
    delta_vx = VX - ux
    
    v_par = delta_vz * Nz + delta_vx * Nx
    v_perp = -delta_vz * Nx + delta_vx * Nz
    
    exponent = -(v_par**2 / v_th_par**2) - (v_perp**2 / v_th_perp**2)
    
    # F. Clipping & Normalization
    pdf_ideal = np.exp(exponent)
    pdf_ideal /= np.max(pdf_ideal)
    
    # Check if peak is even in window (Flux check)
    if np.max(pdf_ideal) < 1e-4:
        return None, None, None, None
    
    # G. Add Noise
    # noise = np.random.normal(0, noise_amp, pdf_ideal.shape)
    # pdf_noisy = np.maximum(pdf_ideal + noise, 0)

    alpha = noise_amp
    sigma = alpha * (1.0 - pdf_ideal)
    noise = np.random.normal(0.0, sigma)

    pdf_noisy = np.maximum(pdf_ideal + noise, 1e-2)
    
    # H. Integrate Moments (The "Ridge")
    # Note: Because vx_grid is hard-limited to +/- 400, 
    # this AUTOMATICALLY simulates transverse clipping!
    flux_profile = np.trapz(pdf_noisy, vx_grid, axis=1)
    
    if np.max(flux_profile) > 0:
        flux_norm = flux_profile / np.max(flux_profile)
    else:
        flux_norm = flux_profile * 0

    numerator = np.trapz(VX * pdf_noisy, vx_grid, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vx_measured = numerator / flux_profile
        
    # I. Prepare Fit Data (Normalized Slope Method)
    # X_data = Vz * sin(theta)
    # NOTE: We pass raw Vz here; the fitter handles the centering.
    X_data = vz_array * Nx
    Y_data = vx_measured    
    
    return X_data, Y_data, flux_norm, theta, VX, VZ, pdf_noisy


if __name__ == "__main__":
    # X, Y, flux, theta, VX, VZ, pdf = generate_numerical_event(2.3, 70, n_steps=100, noise_amp=0.05)
    X, Y, flux, theta, VX, VZ, pdf_noisy = generate_numerical_event_realistic(2.3, 20, n_steps=32, noise_amp=0.05, uz=350, ux=150)

    # VZ = VZ/np.sin(theta)

    # Assuming you have your data unpacked:
    # X_data, Y_data, flux_norm, theta, VX, VZ, pdf_noisy = generate_numerical_event_realistic(...)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Plot the Distribution
    # shading='auto' is usually safer. cmap='turbo' or 'plasma' helps see density.
    c = ax.pcolormesh(VZ, VX, pdf_noisy, cmap='inferno', shading='auto', norm=LogNorm(vmin=1e-2, vmax=1e0))
    plt.colorbar(c, ax=ax, label='Distribution Density')

    # 2. Find the Center of the Distribution (Peak Density)
    # This ensures the arrow starts exactly where the bulk plasma is.
    max_idx = np.unravel_index(np.argmax(pdf_noisy), pdf_noisy.shape)
    center_vz = VZ[max_idx]
    center_vx = VX[max_idx]

    # 3. Define Vector Geometry
    # theta is usually passed in degrees in your loop, convert to radians for math

    # Length of the arrow (in km/s). 200 is a good visual size for solar wind plots.
    arrow_length = 200 

    # Calculate vector components
    # B_z component = cos(theta)
    # B_x component = sin(theta)
    vec_vz = arrow_length * np.cos(theta)
    vec_vx = arrow_length * np.sin(theta)

    # 4. Draw the Vector
    # Using arrow allows for a nice head, but plot is fine too.
    ax.arrow(center_vz, center_vx,      # Start (x, y)
            vec_vz, vec_vx,            # Delta (dx, dy)
            color='white',             # White shows up well on dark colormaps
            width=5,                   # Thickness of the tail
            head_width=20,             # Size of the arrow head
            length_includes_head=True,
            label=r'$\mathbf{B}$ Field Direction')

    # 5. Add a dashed line extending backwards for visual alignment (Optional)
    ax.plot([center_vz - vec_vz, center_vz + vec_vz], 
            [center_vx - vec_vx, center_vx + vec_vx], 
            color='white', linestyle='--', alpha=0.5, zorder=10)

    vzs = VZ[:,0]
    
    sel_vals = np.argmax(pdf_noisy, axis=1)
    vxs = VX[0, sel_vals]

    ax.scatter(vzs, vxs, color='w', marker='o')

    # Labels and Formatting
    ax.set_xlabel(r'$V_z$ (Radial) [km/s]')
    ax.set_ylabel(r'$V_x$ (Transverse) [km/s]')
    ax.set_title(f'Distribution with B-Field Vector (Theta = {np.degrees(theta):.1f}°)')
    ax.set_aspect('equal') # Important! Keeps the angle visually correct
    plt.show()