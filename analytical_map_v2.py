import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def analytical_uncertainty_final():
    # 1. Define Grids (Angle on X-Axis requested)
    angles = np.linspace(-90, 90, 400)
    # Avoid exact 0 and 90 to prevent divide-by-zero
    angles = angles[(np.abs(angles) > 0.5) & (np.abs(angles) < 89.5)]
    
    anisotropies = np.logspace(np.log10(0.03), np.log10(100), 400)
    
    # Meshgrid: X=Angles, Y=Anisotropies
    THETA_DEG, A_TRUE = np.meshgrid(angles, anisotropies)
    theta_rad = np.radians(THETA_DEG)
    Nz = np.cos(theta_rad)
    Nx = np.sin(theta_rad)
    
    # 2. Calculate True Slope 'm' (Ridge Equation)
    # m = ((1-A) * sin * cos) / (cos^2 + A * sin^2)
    # Note: Using standard slope def (Vx vs Vz)
    numerator = (1 - A_TRUE) * Nz * Nx
    denominator = (Nz**2) + A_TRUE * (Nx**2)
    m_true = numerator / denominator
    
    # 3. Dynamic Sigma Model (The "Resolution Penalty")
    # Base error for a "standard" wide distribution
    sigma_base = 0.02 
    
    # Calculate Effective Width along Vz (Look Direction)
    # T_eff_z = cos^2 + A * sin^2 (Normalized to T_par=1)
    T_eff_z = (Nz**2) + A_TRUE * (Nx**2)
    
    # Width W is proportional to sqrt(T). Number of points N is proportional to W.
    # Standard Error of Slope scales as 1/W * 1/sqrt(N) -> 1/W^1.5 ?
    # Let's be conservative and scale linearly with inverse width: 1/sqrt(T)
    # If distribution is 10x narrower, error is ~3x higher.
    
    # We clip the scaling so it doesn't go to infinity, representing min/max resolution
    width_factor = np.sqrt(T_eff_z)
    sigma_dynamic = sigma_base / width_factor
    
    # 4. Perturbation (Calculate Delta+ and Delta-)
    m_upper = m_true + sigma_dynamic
    m_lower = m_true - sigma_dynamic
    
    # Recovery Function
    def recover_A(m, nz, nx):
        denom = (m * (1 - nz**2)) + (nz * nx)
        with np.errstate(divide='ignore', invalid='ignore'):
            epsilon = -m / denom
            A_rec = epsilon + 1 
        return A_rec

    A_rec_1 = recover_A(m_upper, Nz, Nx)
    A_rec_2 = recover_A(m_lower, Nz, Nx)
    
    # 5. Calculate Deltas
    diff1 = A_rec_1 - A_TRUE
    diff2 = A_rec_2 - A_TRUE
    
    delta_plus = np.full_like(A_TRUE, np.nan)
    delta_minus = np.full_like(A_TRUE, np.nan)
    
    for i in range(A_TRUE.shape[0]):
        for j in range(A_TRUE.shape[1]):
            d1 = diff1[i,j]
            d2 = diff2[i,j]
            
            # Max Positive Deviation
            pos_devs = [d for d in [d1, d2] if np.isfinite(d) and d > 0]
            if pos_devs: delta_plus[i,j] = max(pos_devs)
            else: delta_plus[i,j] = 0
            
            # Max Negative Deviation
            neg_devs = [abs(d) for d in [d1, d2] if np.isfinite(d) and d < 0]
            if neg_devs: delta_minus[i,j] = max(neg_devs)
            else: delta_minus[i,j] = 0

    

    # 6. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Delta Plus %
    pct_plus = (delta_plus / A_TRUE) * 100
    # Use pcolormesh with X=THETA, Y=A_TRUE
    c1 = ax[0].pcolormesh(THETA_DEG, A_TRUE, pct_plus, 
                          cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    ax[0].set_yscale('log')
    ax[0].set_title(r'$\Delta_+$ Uncertainty (%) [Measured < True]')
    ax[0].set_xlabel('Incident Angle (deg)')
    ax[0].set_ylabel(r'True Anisotropy ($T_\perp/T_\parallel$)')
    plt.colorbar(c1, ax=ax[0])
    
    # Delta Minus %
    pct_minus = (delta_minus / A_TRUE) * 100
    c2 = ax[1].pcolormesh(THETA_DEG, A_TRUE, pct_minus, 
                          cmap='jet', shading='auto', norm=LogNorm(vmin=1, vmax=100))
    ax[1].set_yscale('log')
    ax[1].set_title(r'$\Delta_-$ Uncertainty (%) [Measured > True]')
    ax[1].set_xlabel('Incident Angle (deg)')
    # Share Y label visually
    plt.colorbar(c2, ax=ax[1])
    
    plt.suptitle("Analytical Uncertainty (Dynamic Resolution Model)")
    plt.tight_layout()
    plt.show()

    return delta_plus, delta_minus, A_TRUE, THETA_DEG

if __name__ == "__main__":
    dp = analytical_uncertainty_final()