import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

# ==========================================
# 1. MONTE CARLO PHYSICS ENGINE
# ==========================================

def generate_monte_carlo_event(velocity_ratio, angle_deg):
    """
    Replicates the particle-based simulation from anisotropy.jy
    """
    # A. Randomize Particle Count (Simulate varying Flux)
    # The code uses: long(rand(1)[0]*1E6) -> Uniform random integer between 0 and 1M
    n_particles = int(np.random.rand() * 1_000_000)
    
    # Avoid empty events
    if n_particles < 100: 
        return np.nan

    # B. Generate Particles (Normalized Thermal Units)
    # p1 = Normal(0, 1) -> Parallel
    # p2 = Normal(0, 1) * aniso -> Perpendicular
    p1 = np.random.randn(n_particles)
    p2 = np.random.randn(n_particles) * velocity_ratio
    
    # C. Geometry
    slope_rad = np.radians(angle_deg)
    Nz = np.cos(slope_rad)
    Nx = np.sin(slope_rad)
    
    # D. Rotate to Instrument Frame (Normalized Vz, Vx)
    # Vz = p1 cos - p2 sin
    # Vx = p2 cos + p1 sin
    vz_particles = p1 * Nz - p2 * Nx
    vx_particles = p1 * Nx + p2 * Nz
    
    # E. Histogram (70x70 bins, Range -5 to 5 sigma)
    # Matches: histogram2d(..., [70,70], dataset([-5,5]), dataset([-5,5]))
    bins = 70
    range_lim = [[-5, 5], [-5, 5]]
    
    hist, x_edges, y_edges = np.histogram2d(vz_particles, vx_particles, 
                                            bins=bins, range=range_lim)
    
    # Get bin centers
    vz_centers = (x_edges[:-1] + x_edges[1:]) / 2
    vx_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # F. Peak Finding (The Ridge)
    # Find max index along Vx axis (axis 1) for each Vz column
    peak_indices = np.argmax(hist, axis=1)
    peak_fluxes = np.max(hist, axis=1)
    
    # Map to Vx values
    vx_peaks = vx_centers[peak_indices]
    
    # G. Fit the Ridge
    # Filter for significant peaks (avoid noise in tails)
    # The Jython code implicitly fits the valid peaks. 
    # We'll use a threshold of 10 particles or 10% of max to be safe.
    threshold = max(5, 0.1 * np.max(peak_fluxes))
    mask = peak_fluxes > threshold
    
    if np.sum(mask) < 4: 
        return np.nan
        
    def linear_model(x, m, c):
        return m * x + c
    
    try:
        # Fit Vz vs Vx peaks
        popt, _ = curve_fit(linear_model, vz_centers[mask], vx_peaks[mask])
        m_fit = popt[0]
    except:
        return np.nan
        
    # H. Inversion (Recover Anisotropy)
    # Formula: epsilon = -m / (m(1-Nz^2) + Nz*Nx)
    denom = (m_fit * (1 - Nz**2)) + (Nz * Nx)
    
    if abs(denom) < 1e-6: return np.nan
    epsilon = -m_fit / denom
    
    if epsilon + 1 < 0: return np.nan
    
    # Recovered Velocity Ratio
    aniso_measured = np.sqrt(epsilon + 1)
    
    return aniso_measured

# ==========================================
# 2. MAP GENERATION LOOP
# ==========================================

def run_simulation_jython_style():
    # Grid Settings from Code [cite: 1]
    # Angles: 0 to 180 (Code says angleMax=180)
    # Anisotropy: 0.01 to 30 (Code says min=0.01, max=30)
    
    angle_bins = np.linspace(0, 180, 60)
    # We assume 'anisotropyList' refers to Temperature Ratio (T_perp/T_par) for the output map axes
    # But the 'aniso' variable in the code is velocity ratio.
    # Let's define the grid in Temperature Ratio units for the map, 
    # but convert to Velocity Ratio for the simulation.
    aniso_ratio_bins = np.logspace(np.log10(0.01), np.log10(30), 40)
    
    delta_plus_map = np.zeros((len(angle_bins), len(aniso_ratio_bins)))
    delta_minus_map = np.zeros((len(angle_bins), len(aniso_ratio_bins)))
    
    print("Running Particle Monte Carlo Simulation...")
    
    for i, angle in enumerate(tqdm(angle_bins)):
        for j, temp_ratio_true in enumerate(aniso_ratio_bins):
            
            # Convert Temperature Ratio to Velocity Ratio for the simulation
            # aniso_vel = sqrt(T_perp / T_par)
            vel_ratio_true = np.sqrt(temp_ratio_true)
            
            diffs = []
            
            # Number of random trials per point (simulating variable flux)
            N_TRIALS = 20 
            
            for _ in range(N_TRIALS):
                # Run the particle simulation
                vel_ratio_meas = generate_monte_carlo_event(vel_ratio_true, angle)
                
                if np.isnan(vel_ratio_meas): continue
                
                # Convert back to Temperature Ratio for error calc?
                # The code calculates: sqrt(epsilon+1) - aniso
                # This implies the error is calculated in VELOCITY space.
                # However, usually Delta Maps are for the final quantity (Temperature Ratio).
                # Let's Stick to the Jython logic: Error in "aniso" parameter (Velocity Ratio).
                # Wait, if the map axes are T_perp/T_par, the error should probably be converted.
                # Let's calculate error in Temperature Ratio to be most useful.
                
                temp_ratio_meas = vel_ratio_meas**2
                diff = temp_ratio_meas - temp_ratio_true
                diffs.append(diff)
            
            # Aggregate Errors
            diffs = np.array(diffs)
            
            # Filter Deltas
            neg_diffs = diffs[diffs < 0] # Measured < True
            pos_diffs = diffs[diffs > 0] # Measured > True
            
            # Store MEAN ABSOLUTE ERROR for this bin
            if len(neg_diffs) > 0:
                delta_plus_map[i, j] = np.mean(np.abs(neg_diffs))
            else:
                delta_plus_map[i, j] = np.nan
                
            if len(pos_diffs) > 0:
                delta_minus_map[i, j] = np.mean(np.abs(pos_diffs))
            else:
                delta_minus_map[i, j] = np.nan

    return angle_bins, aniso_ratio_bins, delta_plus_map, delta_minus_map

# ==========================================
# 3. EXECUTION
# ==========================================

if __name__ == "__main__":
    angles, anisos, d_plus, d_minus = run_simulation_jython_style()
    
    # Save
    np.savez('spc_uncertainty_montecarlo.npz', 
             angles=angles, 
             anisotropies=anisos, 
             delta_plus=d_plus, 
             delta_minus=d_minus)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    X, Y = np.meshgrid(anisos, angles)
    
    # Normalize error by the True Value for % plot
    pct_plus = (d_plus / X) * 100
    pct_minus = (d_minus / X) * 100
    
    c1 = ax[0].pcolormesh(X, Y, pct_plus, cmap='jet', shading='auto', vmin=0, vmax=100)
    ax[0].set_xscale('log')
    ax[0].set_title('Delta Plus % (Measured < True)')
    ax[0].set_ylabel('Incident Angle (deg)')
    ax[0].set_xlabel('True Anisotropy ($T_\perp/T_\parallel$)')
    plt.colorbar(c1, ax=ax[0])
    
    c2 = ax[1].pcolormesh(X, Y, pct_minus, cmap='jet', shading='auto', vmin=0, vmax=100)
    ax[1].set_xscale('log')
    ax[1].set_title('Delta Minus % (Measured > True)')
    ax[1].set_xlabel('True Anisotropy ($T_\perp/T_\parallel$)')
    plt.colorbar(c2, ax=ax[1])
    
    plt.suptitle("Uncertainty Map (Particle Monte Carlo Method)")
    plt.tight_layout()
    plt.show()