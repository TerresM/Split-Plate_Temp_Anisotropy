import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

def generate_analytical_map():
    # 1. Define the Parameter Space
    # We sample the same grid as the MCMC
    angles = np.linspace(-90, 90, 1000)
    # Avoid the exact singularities at 0 and 90
    angles = angles[(np.abs(angles) > 1) & (np.abs(angles) < 89)]
    
    anisotropies = np.logspace(np.log10(0.03), np.log10(30), 500)
    
    # Create 2D Grid
    A_TRUE, THETA_DEG = np.meshgrid(anisotropies, angles)
    theta_rad = np.radians(THETA_DEG)
    Nz = np.cos(theta_rad)
    Nx = np.sin(theta_rad)

    # 2. Calculate the "True" Slope for every point
    # Using the standard Vz vs Vx slope formula (matching the Jython logic)
    # m = (1-A)*cos*sin / (cos^2 + A*sin^2) ... wait, let's use the inverse relation derived previously.
    # Actually, let's just use the Forward Relation derived from the Ridge Equation:
    # m_true = (Nx * Nz * (1 - A_TRUE)) / ((Nz**2) + A_TRUE * (Nx**2))
    # This assumes the Standard Slope fit (Vx vs Vz)
    
    # Let's double check the Jython inversion: epsilon = -m / (m(1-Nz^2) + Nz*Nx)
    # This implies the forward relation for m is:
    # m = -epsilon * Nz * Nx / (1 + epsilon*(1-Nz^2))
    # Since A = sqrt(epsilon+1), then epsilon = A^2 - 1.
    # Let's stick to the inversion formula which is the ground truth for recovery.
    
    # Let's calculate the slope 'm' that would result in the True Anisotropy.
    # Invert the Jython formula: m = (Nz*Nx * (1-A^2)) / (A^2 * Nx^2 + Nz^2)
    # Wait, let's rely on the recovery function directly.
    
    # REVERSE ENGINEERED FORWARD FUNCTION (Vx vs Vz standard slope)
    # If A_true is known, the ideal slope m_true is:
    # m_true = ( (1 - A_TRUE**2) * Nz * Nx ) / ( (A_TRUE**2 * Nx**2) + Nz**2 )
    # Note: Jython code uses 'aniso' as velocity ratio, so A = aniso^2 here if aniso is temp ratio.
    # Let's assume A_TRUE is Temperature Anisotropy (T_perp/T_par).
    
    m_true = ( (1 - A_TRUE) * Nz * Nx ) / ( A_TRUE * Nx**2 + Nz**2 )

    # 3. Define the Instrument Noise (Sigma_m)
    # This is the ONLY "fudge factor". It represents how messy the ridge fit gets.
    # A value of 0.02 means the slope is uncertain by +/- 0.02 (dimensionless).
    # You can make this a function of Angle if you want to simulate clipping.
    sigma_m = 0.03 
    
    # 4. Calculate the Bounds (Propagate Error)
    m_upper = m_true + sigma_m
    m_lower = m_true - sigma_m
    
    # 5. Recovery Function (The Analytic Inversion)
    def recover_A(m, nz, nx):
        denom = (m * (1 - nz**2)) + (nz * nx)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            epsilon = -m / denom
            A_rec = epsilon + 1 # This is T_perp/T_par
        return A_rec

    # Recover Anisotropy for the perturbed slopes
    A_rec_1 = recover_A(m_upper, Nz, Nx)
    A_rec_2 = recover_A(m_lower, Nz, Nx)
    
    # 6. Calculate Delta Plus / Delta Minus
    # Because the function flips (derivative changes sign), we check max/min deviations
    
    # Difference from Truth
    diff1 = A_rec_1 - A_TRUE
    diff2 = A_rec_2 - A_TRUE
    
    # We want to know: "How far could I be OFF in the positive direction?"
    # And "How far could I be OFF in the negative direction?"
    
    # Initialize arrays filled with NaNs (for regions where recovery fails)
    delta_plus = np.full_like(A_TRUE, np.nan)
    delta_minus = np.full_like(A_TRUE, np.nan)
    
    # Logic: 
    # If diff1 is +0.5 and diff2 is -0.2, then Delta+ is 0.5, Delta- is 0.2
    # If both are positive (rare, implies non-monotonic), take max.
    
    # Vectorized assignment
    for i in range(A_TRUE.shape[0]):
        for j in range(A_TRUE.shape[1]):
            d1 = diff1[i,j]
            d2 = diff2[i,j]
            
            # Collect valid positive deviations (A_rec > A_true)
            pos_devs = [d for d in [d1, d2] if np.isfinite(d) and d > 0]
            if pos_devs:
                delta_plus[i,j] = max(pos_devs)
            else:
                delta_plus[i,j] = 0.0 # or nan
                
            # Collect valid negative deviations (A_rec < A_true)
            neg_devs = [abs(d) for d in [d1, d2] if np.isfinite(d) and d < 0]
            if neg_devs:
                delta_minus[i,j] = max(neg_devs)
            else:
                delta_minus[i,j] = 0.0 # or nan

    # 7. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Delta Plus %
    pct_plus = (delta_plus / A_TRUE) * 100
    c1 = ax[0].pcolormesh( THETA_DEG, A_TRUE, pct_plus, cmap='jet', shading='auto', norm=LogNorm(vmin=1e0, vmax=1e2))#, vmin=0, vmax=100)
    ax[0].set_yscale('log')
    ax[0].set_title(r'Analytical $\Delta_+$ Uncertainty (%)')
    ax[0].set_xlabel('Incident Angle (deg)')
    ax[0].set_ylabel('True Anisotropy ($T_\perp/T_\parallel$)')
    plt.colorbar(c1, ax=ax[0])

    # Delta Minus %
    pct_minus = (delta_minus / A_TRUE) * 100
    c2 = ax[1].pcolormesh( THETA_DEG, A_TRUE, pct_minus, cmap='jet', shading='auto', norm=LogNorm(vmin=1e0, vmax=1e2))#, vmin=0, vmax=100)
    ax[1].set_yscale('log')
    ax[1].set_title(r'Analytical $\Delta_-$ Uncertainty (%)')
    # ax[1].set_ylabel('True Anisotropy ($T_\perp/T_\parallel$)')
    ax[1].set_xlabel('Incident Angle (deg)')
    plt.colorbar(c2, ax=ax[1])
    
    plt.suptitle(f"Analytical Uncertainty Model (Assumed Slope Noise $\sigma_m={sigma_m}$)")
    plt.tight_layout()
    plt.show()

    # 7. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Delta Plus %
    pct_plus = (delta_plus / A_TRUE) * 100
    c1 = ax[0].pcolormesh( THETA_DEG, A_TRUE, delta_plus, cmap='jet', shading='auto', norm=LogNorm(vmin=1e0, vmax=1e2))#, vmin=0, vmax=100)
    ax[0].set_yscale('log')
    ax[0].set_title(r'Analytical $\Delta_+$ Uncertainty (%)')
    ax[0].set_ylabel('Incident Angle (deg)')
    ax[0].set_xlabel('True Anisotropy ($T_\perp/T_\parallel$)')
    plt.colorbar(c1, ax=ax[0])

    # Delta Minus %
    pct_minus = (delta_minus / A_TRUE) * 100
    c2 = ax[1].pcolormesh( THETA_DEG, A_TRUE, delta_minus, cmap='jet', shading='auto', norm=LogNorm(vmin=1e0, vmax=1e2))#, vmin=0, vmax=100)
    ax[1].set_yscale('log')
    ax[1].set_title(r'Analytical $\Delta_-$ Uncertainty (%)')
    ax[1].set_xlabel('True Anisotropy ($T_\perp/T_\parallel$)')
    plt.colorbar(c2, ax=ax[1])
    
    plt.suptitle(f"Analytical Uncertainty Model (Assumed Slope Noise $\sigma_m={sigma_m}$)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_analytical_map()