import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

def analytical_uncertainty_tuned():
    # 1. Grids
    angles = np.linspace(0.1, 89.9, 500)
    anisotropies = np.logspace(np.log10(0.05), np.log10(30), 500) # Match simulation range
    
    THETA_DEG, A_TRUE = np.meshgrid(angles, anisotropies)
    theta_rad = np.radians(THETA_DEG)
    S = np.sin(theta_rad)
    C = np.cos(theta_rad)
    
    # 2. Sensitivity (The Geometry)
    # This creates the vertical Red stripes at 0 and 90
    denom_z = (C**2) + A_TRUE * (S**2)
    dm_dA = -1.0 * (S * C) / (denom_z**2)
    sensitivity = np.abs(dm_dA)
    
    # 3. Noise Model Tuning
    sigma_base = 0.025  # Slightly higher base noise to match simulation contrast
    
    # A. Resolution Penalty (Low Anisotropy / Cold distributions)
    # This creates the Horizontal Red stripe at the bottom
    resolution_penalty = 1.0 / np.sqrt(denom_z)
    
    # B. Aperture Clipping Penalty (High Anisotropy)
    # T_x is the Transverse Width (projected beam width)
    # At 0 deg, T_x = A (Full Perpendicular Temp)
    T_x = (S**2) + A_TRUE * (C**2)
    
    # --- TUNING PARAMETERS ---
    # Previous: T_limit=5, Exponent=4 (Too Harsh)
    # New: T_limit=20, Exponent=2 (Softer)
    # This means clipping only gets "bad" when the transverse width is > 20x the parallel width.
    T_limit = 15.0 
    clipping_penalty = 1.0 + (T_x / T_limit)**4
    
    # Combine
    sigma_m = sigma_base * resolution_penalty * clipping_penalty
    
    # 4. Uncertainty Calculation
    sigma_A = sigma_m / (sensitivity + 1e-9)
    delta_rel = (sigma_A / A_TRUE) * 100
    delta_plot = np.clip(delta_rel, 1, 100) # Clip for visualization

    # 5. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: The Simulation Match
    # Using pcolormesh to replicate the simulation look
    c1 = ax[0].pcolormesh(THETA_DEG, A_TRUE, delta_plot, 
                          cmap='jet', shading='auto', 
                          norm=LogNorm(vmin=1, vmax=100))
    ax[0].set_yscale('log')
    ax[0].set_title(r'Analytical Model (Tuned to Simulation)')
    ax[0].set_xlabel('Incident Angle (deg)')
    ax[0].set_ylabel(r'True Anisotropy ($T_\perp/T_\parallel$)')
    
    # Add negative angles to match simulation plot style (Mirroring)
    # This is just for visual comparison
    ax[0].set_xlim(0, 90)
    
    # Plot 2: Full mirrored view to compare with your image
    # We concatenate to make it -90 to 90
    full_angles = np.concatenate([-np.flip(angles), angles])
    full_delta = np.concatenate([np.flip(delta_plot, axis=1), delta_plot], axis=1)
    # Re-mesh for full plot
    FULL_THETA, FULL_A = np.meshgrid(full_angles, anisotropies)
    
    c2 = ax[1].pcolormesh(FULL_THETA, FULL_A, full_delta, 
                          cmap='jet', shading='auto', 
                          norm=LogNorm(vmin=1, vmax=100))
    ax[1].set_yscale('log')
    ax[1].set_title(r'Full Range Comparison (-90 to 90)')
    ax[1].set_xlabel('Incident Angle (deg)')
    
    plt.colorbar(c2, ax=ax[1], label='Percent Uncertainty')
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def generate_final_delta_plots():
    # 1. Define Grids (0 to 90)
    # We simulate one quadrant and mirror it for perfect symmetry
    angles = np.linspace(0.1, 89.9, 500)
    anisotropies = np.logspace(np.log10(0.05), np.log10(30), 500)
    
    THETA_DEG, A_TRUE = np.meshgrid(angles, anisotropies)
    theta_rad = np.radians(THETA_DEG)
    S = np.sin(theta_rad)
    C = np.cos(theta_rad)
    
    # 2. Geometric Sensitivity (dm/dA)
    # The derivative of the ridge curve
    denom_z = (C**2) + A_TRUE * (S**2)
    dm_dA = -1.0 * (S * C) / (denom_z**2)
    sensitivity = np.abs(dm_dA)
    
    # 3. Tuned Noise Model
    sigma_base = 0.04
    
    # A. Resolution Penalty (Low Anisotropy)
    # Penalizes narrow distributions (cold T_z)
    resolution_penalty = 1.0 / np.sqrt(denom_z)
    
    # B. Aperture Clipping Penalty (High Anisotropy)
    # Transverse Width T_x
    T_x = (S**2) + A_TRUE * (C**2)
    
    # TUNED PARAMETERS (User Request)
    T_limit = 5.0 
    exponent = 4.0
    
    # The Penalty Function
    clipping_penalty = 1.0 + (T_x / T_limit)**exponent
    
    # Total Noise Sigma
    sigma_m = sigma_base * resolution_penalty * clipping_penalty
    
    # 4. Calculate Relative Uncertainty (%)
    sigma_A = sigma_m / (sensitivity + 1e-9)
    delta_rel = (sigma_A / A_TRUE) * 100
    
    # Clip for clean plotting (1% to 100%)
    delta_plot = np.clip(delta_rel, 1, 100)
    
    # 5. Mirror Data for -90 to 90 Plot
    # Concatenate [Flip(-Angles), Angles]
    full_angles = np.concatenate([-np.flip(angles), angles])
    # Concatenate [Flip(Data), Data]
    full_delta = np.concatenate([np.flip(delta_plot, axis=1), delta_plot], axis=1)

    full_delta2 = np.concatenate([np.flip(sigma_A, axis=1), sigma_A], axis=1)
    
    # Create new meshgrid for the full range
    FULL_THETA, FULL_A = np.meshgrid(full_angles, anisotropies)

    # 6. Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    
    # PLOT 1: Delta Plus
    # (In analytical derivative models, Delta+ approx Delta-)
    c1 = ax[0].pcolormesh(FULL_THETA, FULL_A, full_delta, 
                          cmap='jet', shading='auto', 
                          norm=LogNorm(vmin=1, vmax=100))
    ax[0].set_yscale('log')
    ax[0].set_title(r'Percent Uncertainty Upper Bound ($\Delta_+$)', fontsize=14)
    ax[0].set_ylabel(r'True Anisotropy ($T_\perp/T_\parallel$)', fontsize=12)
    ax[0].tick_params(labelbottom=False) # Hide x-labels on top plot
    plt.colorbar(c1, ax=ax[0], label='% Uncertainty')
    
    # PLOT 2: Delta Minus
    c2 = ax[1].pcolormesh(FULL_THETA, FULL_A, full_delta2, 
                          cmap='jet', shading='auto', 
                          norm=LogNorm(vmin=1, vmax=100))
    ax[1].set_yscale('log')
    ax[1].set_title(r'Percent Uncertainty Lower Bound ($\Delta_-$)', fontsize=14)
    ax[1].set_ylabel(r'True Anisotropy ($T_\perp/T_\parallel$)', fontsize=12)
    ax[1].set_xlabel('Incident Magnetic Field Angle (deg)', fontsize=12)
    plt.colorbar(c2, ax=ax[1], label='% Uncertainty')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_final_delta_plots()

# if __name__ == "__main__":
#     analytical_uncertainty_tuned()