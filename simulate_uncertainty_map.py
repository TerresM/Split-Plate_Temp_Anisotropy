import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

def linear_model(x, a, b):
    """Simple linear model: y = ax + b"""
    return a * x + b

def find_slopes(model, xdata, ydata, weights, method='lm'):
    """
    Performs linear fitting using scipy.optimize.curve_fit.
    Matches the logic in spc_calc_anisotropy_v3.py
    
    Parameters:
    - weights: This corresponds to 'sigma' in curve_fit (uncertainty), NOT 1/sigma^2.
               In the pipeline, this is passed as 1.0/rdf_weights.
    """
    # Perform the fitting
    # Note: absolute_sigma=True is generally required if weights are real sigmas, 
    # but we stick to the default (False) if not specified in your file to match exact behavior.
    popt, pcov = curve_fit(model, xdata, ydata, sigma=weights, method=method)

    # Calculate statistics (needed for uncertainty in the pipeline, but here we just need popt)
    linear_fit = model(xdata, *popt)
    
    # We return just what is needed for the simulation
    return popt

def _calc_anisotropy_from_fits(fit_value, NZ):
    """
    Inverts the slope to find anisotropy (T_ani).
    Equation: slope = (1-A) / ( (1-Nz^2) + Nz/slope_factor? ) ... 
    Based on code: alpha = (-m / (m * (1 - Nz^2) + Nz)) + 1
    """
    slope = fit_value[0]
    
    # Formula from your script:
    # alpha = -(chi / (chi (1 - Nz^2) + Nz)) + 1
    # where chi is the slope.
    denominator = slope * (1. - NZ**2) + NZ
    
    if abs(denominator) < 1e-9:
        return np.nan, np.nan
        
    alpha = (-slope / denominator) + 1

    if np.isfinite(alpha) and (alpha > 0):
        T_ani = np.sqrt(alpha)**2 # Ensures positive, though sqrt(alpha^2) == alpha if > 0
    else:
        T_ani = np.nan
    
    return alpha, T_ani

# ==========================================
# 2. SIMULATION LOGIC
# ==========================================

# IDK why this function is being created. But I will see. 
def theoretical_slope_from_alpha(alpha, Nz):
    """
    Inverse of the _calc_anisotropy_from_fits function to generate ground truth.
    alpha = (-m / (m * (1 - Nz^2) + Nz)) + 1
    alpha - 1 = -m / (m(1-Nz^2) + Nz)
    (alpha - 1) * [m(1-Nz^2) + Nz] = -m
    m(alpha-1)(1-Nz^2) + Nz(alpha-1) = -m
    m [ (alpha-1)(1-Nz^2) + 1 ] = -Nz(alpha-1)
    m = -Nz(alpha-1) / [ (alpha-1)(1-Nz^2) + 1 ]
    m = Nz(1-alpha) / [ (alpha-1)sin^2(theta) + 1 ]
    """
    sin2 = 1.0 - Nz**2
    numerator = Nz * (1.0 - alpha)
    denominator = (alpha - 1.0) * sin2 + 1.0
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

def run_uncertainty_simulation(NANI, NTH, L=16):

    # Configuration
    N_RUNS = 500    # The number of runs to use per pixel.
                    # This includes new random noise instances.
    THRESHOLD = 0.5

    # Define the SPC model like parameters
    PEAK_CURRENT  = 30.0   # pA: This value is typical of the solar wind.
    NOISE_CURRENT = 1.0     # pA: This is the defined average noise value. 
                            # Note: we can make this velocity dependent sigma_v = sigma_I/v^2.

    # Define the SPC VZ grid
    vz_spc = np.geomspace(138, 875, L)     # 32 voltage windows. This should be adjusted to match
                                            # the peak track windows. 
    # vz_spc = np.geomspace(250, 550, L)

    # Define the values we are solving for
    delta_plus = np.zeros((NANI, NTH))
    delta_minus = np.zeros((NANI, NTH))

    # Define the preset SPC model parameters
    # NOTE: We should likely define this uncertainty as a function of Mach Number. V/V_th
    v_bulk    = 400    # Moderate solar wind speed
    v_th_para = 40     # Define a moderate parallel thermal speed

    # Define the range of anisotropy and magnetic field angles we are using. 
    ani_vals = np.logspace(np.log10(0.05), np.log10(30.0), NANI)
    th_vals  = np.linspace(0.5, 89.5, NTH)

    rdfs = np.zeros((NANI, NTH, L, N_RUNS))
    vxs  = np.zeros((NANI, NTH, L, N_RUNS))
    for i, ani_true in enumerate(ani_vals):
        for j, theta_true in enumerate(th_vals):
            # Geometry of the background magnetic field
            theta_rad = np.radians(theta_true)
            Nz = np.cos(theta_rad)
            Nx = np.sin(theta_rad)

            # Define the effective temperature
            T_eff_norm = (Nz**2) + ani_true * (Nx**2)
            v_th_eff = v_th_para * np.sqrt(T_eff_norm)

            # print(f"The effective thermal speed is {v_th_eff}")

            # Define the ideal reduced distirbution function
            current = np.exp(-1.0 * ((vz_spc - v_bulk)/v_th_eff)**2)

            current_ideal = PEAK_CURRENT * current
            current_noise = np.random.normal(0, 1, (L, N_RUNS)) * NOISE_CURRENT

            current_model = (current_ideal[:,None] + current_noise) + 1.0

            # Convert to RDF
            rdf_model = current_model / (vz_spc[:,None]**2)
            rdf_model = rdf_model/np.max(rdf_model)

            rdfs[i,j,:,:] = rdf_model

            # Define the vx_direction
            # m = (1 - A) Nx Nz / (Nz^2 + A Nx^2)
            # Notice, we are moving the sin theta = Nx, to the RHS
            # This implies vx = m vz.
            m_ideal  = ((1 - ani_true) * np.cos(theta_rad)) / (np.cos(theta_rad)**2 + ani_true * np.sin(theta_rad)**2)
            vx_ideal = m_ideal * vz_spc * np.sin(theta_rad)

            # plt.figure(); plt.scatter(vz_spc * np.sin(theta_rad), vx_ideal); plt.ylim(0, 1200)

            sigma_vx = 2.0 * (NOISE_CURRENT / (current_ideal + 0.01))    # Do not divide by zero

            vx_noise = np.random.normal(0, 1, (L, N_RUNS)) * sigma_vx[:,None]
            vx_model = vx_ideal[:,None] + vx_noise

            # plt.scatter(vz_spc * np.sin(theta_rad), vx_model[:,0])
            vx_model = vx_ideal[:,None] + vx_noise
            xdata = vz_spc * np.sin(theta_rad)

            vxs[i,j,:,:] = vx_model

            results = []
            for k in range(N_RUNS):
                # Calculate Weights from Noisy RDF
                r = rdf_model[:,k]
                r_max = np.max(r)
                
                if r_max <= 0: continue
                
                weights = r / r_max
                
                # Apply Threshold Mask
                mask = (weights > THRESHOLD)
                if np.sum(mask) < 3: continue
                
                # Fit Slope to Noisy Vx using Noisy Weights
                popt = find_slopes(linear_model, xdata[mask], vx_model[:,k][mask], weights[mask])
                # print(popt)
                # input('What is your quest?')
                # Invert
                alpha, Tani = _calc_anisotropy_from_fits(popt, Nz)

                # print(Tani, ani_true)
                # print(alpha, Tani)
                # input('What is your favorite color?')
                if np.isfinite(Tani):
                    results.append(Tani)
            
            # 5. Statistics
            if len(results) > 10:
                res = np.array(results)
                delta_plus[i,j] = np.percentile(res, 84) - ani_true
                delta_minus[i,j] = ani_true - np.percentile(res, 16)
            else:
                delta_plus[i,j] = np.nan
                delta_minus[i,j] = np.nan
        # plt.show()
        if i % 5 == 0: print(f"Row {i}/{NANI} done...")          
    plot_simulation_results(th_vals, ani_vals, delta_plus, delta_minus)

    return(ani_vals, th_vals, vz_spc, rdfs, vxs)


def plot_simulation_results(angles, anisotropies, d_plus, d_minus):
    X, Y = np.meshgrid(angles, anisotropies)
    
    # Calc percent
    dp_pct = (d_plus / Y) * 100
    dm_pct = (d_minus / Y) * 100
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    c1 = ax[0].pcolormesh(X, Y, dp_pct, cmap='jet', shading='auto', 
                          norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=100))
    ax[0].set_yscale('log')
    ax[0].set_title(r'Upper Bound Uncertainty ($\Delta_+$ %)')
    ax[0].set_ylabel(r'True Anisotropy ($T_\perp/T_\parallel$)')
    ax[0].set_xlabel('Angle (deg)')
    plt.colorbar(c1, ax=ax[0])
    
    c2 = ax[1].pcolormesh(X, Y, dm_pct, cmap='jet', shading='auto', 
                          norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=100))
    ax[1].set_yscale('log')
    ax[1].set_title(r'Lower Bound Uncertainty ($\Delta_-$ %)')
    ax[1].set_xlabel('Angle (deg)')
    plt.colorbar(c2, ax=ax[1])
    
    plt.suptitle('SPC Simulation using Pipeline Fitting Logic', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run_pipeline_simulation()
    ani_vals, th_vals, a, b, c = run_uncertainty_simulation(50, 50, L = 32)