import numpy as np
import cdflib.xarray
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

plt.ion()

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

def build_clean_2d_state(vx_grid, vz_grid, sample, theta_bn, R_true):
    """
    This function constructs a 2D bi-Maxwellian, drawn from a sample of measurements 
    made from SPC from Encounters 11 to 23.

    Inputs:
    -------
    sample : list
        Contains the density, velocity, and temperature information.
    theta_bn : float
        Angle (in degrees) between the magnetic field and the x-z plane. 
    R_thre : float
        The true temperature anisotropy.

    Outputs:
    --------
    """
    # Get the samples
    n_sample = sample[0]
    ux_sample = sample[1]
    uz_sample = sample[2]
    w_sample  = sample[3]

    # Define the known values
    theta_b = np.radians(theta_bn)
    bz = np.cos(theta_b)
    bx = np.sin(theta_b)

    # Derive the parallel and perpendicular thermal speed
    w_para = w_sample/np.sqrt(bz**2 + R_true * (1 - bz**2))
    w_perp = np.sqrt(R_true) * w_para

    # Define the vpara and vperp
    dv_x = vx_grid - ux_sample
    dv_z = vz_grid - uz_sample

    # Define the parallel component of the distribution
    v_para = (dv_x * bx) + (dv_z * bz)
    v_perp_sq = (dv_z * bx - dv_x * bz)**2

    # Define the exponentials
    exp_para = np.exp(-(v_para**2)/(w_para**2))
    exp_perp = np.exp(-(v_perp_sq)/(w_perp**2))

    # Define the normalization
    norm = n_sample/(np.pi * w_perp * w_para)

    f_2d = norm * exp_para * exp_perp

    return(f_2d)


def build_1d_vdf_clean(vx_grid, vz_grid, sample, theta_bn, R_true):
    """
    This function constructs a 2D bi-Maxwellian, drawn from a sample of measurements 
    made from SPC from Encounters 11 to 23.

    Inputs:
    -------
    sample : list
        Contains the density, velocity, and temperature information.
    theta_bn : float
        Angle (in degrees) between the magnetic field and the x-z plane. 
    R_thre : float
        The true temperature anisotropy.

    Outputs:
    --------
    """
    # Get the samples
    n_sample = sample[0]
    ux_sample = sample[1]
    uz_sample = sample[2]
    w_sample  = sample[3]

    # Define the known values
    theta_b = np.radians(theta_bn)
    bz = np.cos(theta_b)
    bx = np.sin(theta_b)

    # Derive the parallel and perpendicular thermal speed
    w_para = w_sample/np.sqrt(bz**2 + R_true * (1 - bz**2))
    w_perp = np.sqrt(R_true) * w_para

    # Define the vpara and vperp
    dv_x = vx_grid - ux_sample
    dv_z = vz_grid - uz_sample

    # Define the parallel component of the distribution
    v_para = (dv_x * bx) + (dv_z * bz)
    v_perp_sq = (dv_z * bx - dv_x * bz)**2

    # Define the exponentials
    exp_para = np.exp(-(v_para**2)/(w_para**2))
    exp_perp = np.exp(-(v_perp_sq)/(w_perp**2))

    # Define the normalization
    norm = n_sample/(np.pi * w_perp * w_para)

    f_2d = norm * exp_para * exp_perp

    # Define the FOV mask
    incident_angles = np.abs(np.arctan2(vx_grid, np.abs(vz_grid)))
    FOV_mask = incident_angles <= np.radians(30.0)      # Only accept up to the instrument half angle.

    f_2d_masked = f_2d * FOV_mask

    # Get the 1D reduced VDF
    F_1D = np.trapezoid(f_2d_masked, x=vx_array, axis=0)

    return(F_1D)

def build_1d_vdf_model(F_1D, vz_array):
    A_eff = 1.59
    q_e   = 1.602e-19
    C_cal = 1e5 * 1e12 
    dvz_array = np.gradient(vz_array)
    conversion_factor = A_eff * q_e * np.abs(vz_array) * dvz_array * C_cal

    I_ideal = F_1D * conversion_factor  # This is a proxy for the current.

    # We assume there is a 1 pA noise floor
    I_noise_bg   = np.random.normal(0, 1, size = len(vz_array))

    I_model = I_ideal + I_noise_bg + 1.0

    # Define the reduced vdf model
    F_1D_model = I_model/ conversion_factor
    F_1D_model = np.clip(F_1D_model, 0, None)

    return(F_1D_model, I_ideal)

def build_1d_vdf_clean_vectorized(VX, VZ, n_arr, ux_arr, uz_arr, wp_arr, theta_bn, R_true, FOV_mask, vx_array):
    """
    Fully vectorized VDF builder. Calculates NRUNS simultaneously.
    """
    # Define the known values
    theta_b = np.radians(theta_bn)
    bz = np.cos(theta_b)
    bx = np.sin(theta_b)

    # Derive the parallel and perpendicular thermal speed arrays
    w_para_arr = wp_arr / np.sqrt(bz**2 + R_true * (1 - bz**2))
    w_perp_arr = np.sqrt(R_true) * w_para_arr

    # Reshape the 1D sample arrays (NRUNS,) to 3D (NRUNS, 1, 1) for broadcasting
    n_3d = n_arr[:, None, None]
    ux_3d = ux_arr[:, None, None]
    uz_3d = uz_arr[:, None, None]
    w_para_3d = w_para_arr[:, None, None]
    w_perp_3d = w_perp_arr[:, None, None]

    # Reshape VX and VZ from (Nx, Nz) to (1, Nx, Nz)
    VX_3d = VX[None, :, :]
    VZ_3d = VZ[None, :, :]

    # Calculate kinematics for all 10,000 runs simultaneously
    dv_x = VX_3d - ux_3d
    dv_z = VZ_3d - uz_3d

    v_para = (dv_x * bx) + (dv_z * bz)
    v_perp_sq = (dv_z * bx - dv_x * bz)**2

    # Define the exponentials
    # exp_para = np.exp(-(v_para**2) / (w_para_3d**2))
    # exp_perp = np.exp(-(v_perp_sq) / (w_perp_3d**2))
    exponent = np.exp((-(v_para**2) / (w_para_3d**2)) + (-(v_perp_sq) / (w_perp_3d**2)))

    # Define the normalization
    norm = n_3d / (np.pi * w_perp_3d * w_para_3d)

    # Create the full 3D array of VDFs (Shape: NRUNS, Nx, Nz)
    # f_2d = norm * exp_para * exp_perp
    f_2d = norm * exponent
    # Apply the pre-calculated FOV mask (Broadcasts across NRUNS)
    f_2d_masked = f_2d * FOV_mask[None, :, :]

    # Integrate across the vx_array (which is axis 1 in our 3D setup)
    F_1D_all = np.trapezoid(f_2d_masked, x=vx_array, axis=1)

    return F_1D_all  # Shape is perfectly (NRUNS, Nz)

def build_1d_vdf_model_vectoriezed(f_1ds, vz_array):
    A_eff = 1.59
    q_e   = 1.602e-19
    C_cal = 1e5 * 1e12 
    dvz_array = np.gradient(vz_array)
    conversion_factor = A_eff * q_e * np.abs(vz_array) * dvz_array * C_cal

    I_ideal = f_1ds * conversion_factor  # This is a proxy for the current.

    I_noise_bg = np.random.normal(0, 1, size=I_ideal.shape)

    I_model = I_ideal + I_noise_bg + 1.0

    # Define the reduced vdf model
    f_1ds_models = I_model/conversion_factor
    f_1ds_models = np.clip(f_1ds_models, 0, None)

    return f_1ds_models, I_ideal

if __name__ == "__main__":
    NTH, NANI = 50, 50

    # Load in the samples, let us start with the moments.
    samples = cdflib.xarray.cdf_to_xarray('spc_all_valid_moms.cdf')

    sample_mask = (samples.np_moment.data > 10) & (samples.wp_moment.data > 20)

    samples = samples.isel(Epoch=sample_mask)

    # High resolution grid, this is to be comparable to peak tracks
    speed_bins = np.logspace(np.log10(138), np.log10(875), num=128) # Assume 60 energy windows
    vz_array = np.sort(speed_bins)

    vx_limit = np.max(np.abs(vz_array)) * np.tan(np.radians(30))
    vx_array = np.linspace(-vx_limit, vx_limit, 60)

    VX, VZ = np.meshgrid(vx_array, vz_array, indexing='ij')

    incident_angles = np.abs(np.arctan2(VX, np.abs(VZ)))
    FOV_mask = incident_angles <= np.radians(30.0)    

    theta_vals = np.linspace(0.3, 89.5, NTH)
    ani_vals = np.logspace(np.log10(0.05), np.log10(30.0), NANI)

    # Define the values we are solving for
    delta_plus = np.zeros((NANI, NTH))
    delta_minus = np.zeros((NANI, NTH))

    NRUNS = 10000

    # Define the total number of idxs
    idxs = np.random.randint(0, len(samples.Epoch.data), size=(NRUNS))

    n_arr = samples.np_moment.values[idxs]
    ux_arr = samples.vp_moment_SC.values[idxs,0]
    uz_arr = -samples.vp_moment_SC.values[idxs,2]
    wp_arr = samples.wp_moment.values[idxs]

    # for i, alpha in tqdm(enumerate(ani_vals)):
    for i in tqdm(range(25,27)):
        alpha = ani_vals[i]
        for j in range(50):
        # for j, theta_b in enumerate(theta_vals):
            theta_b = np.radians(theta_vals[j])
            # alpha = ani_vals[10]
            # alpha = 1.2

            f_1ds = build_1d_vdf_clean_vectorized(VX, VZ, n_arr, ux_arr, uz_arr, wp_arr, theta_b, alpha, FOV_mask, vx_array)

            f_1ds_models, I_ideal = build_1d_vdf_model_vectoriezed(f_1ds, vz_array)

            m_ideal  = ((1 - alpha) * np.cos(theta_b)) / (np.cos(theta_b)**2 + alpha * np.sin(theta_b)**2)
            vx_ideal = m_ideal * vz_array * np.sin(theta_b)

            sigma_vx = 5.0 * (1.0 / np.sqrt(I_ideal + 0.01))
            
            vx_noise = np.random.normal(0, 1, (len(vz_array))) * sigma_vx
            vx_model = vx_ideal + vx_noise

            xdata = vz_array * np.sin(theta_b)
            weights = f_1ds_models/np.max(f_1ds_models, axis=1)[:,None]

            mask = weights > 0.5

            results = []
            slopes = []
            
            # 1. Setup X, Y, and Weights for Vectorization
            X = xdata[None, :]   # Broadcast to shape (1, len(vz_array))
            Y = vx_model         # Shape (NRUNS, len(vz_array))
            
            # Initialize weights. In curve_fit, sigma represents uncertainty. 
            # For Weighted Least Squares, W = 1 / sigma^2. 
            # We zero out weights where the mask is False to ignore those points.
            W = np.zeros_like(Y)
            W[mask] = 1.0 / (weights[mask]**2)
            
            # 2. Vectorized Sums across the velocity axis (axis=1)
            S_w   = np.sum(W, axis=1)
            S_wx  = np.sum(W * X, axis=1)
            S_wy  = np.sum(W * Y, axis=1)
            S_wxx = np.sum(W * X * X, axis=1)
            S_wxy = np.sum(W * X * Y, axis=1)
            
            # 3. Calculate Determinant
            delta = (S_w * S_wxx) - (S_wx**2)
            
            # 4. Calculate Slopes Safely
            # Ensure we have enough points (>1) and a non-zero determinant
            valid_fits = (np.sum(mask, axis=1) > 1) & (np.abs(delta) > 1e-12)
            
            slopes = np.full(NRUNS, np.nan)
            slopes[valid_fits] = ((S_w[valid_fits] * S_wxy[valid_fits]) - (S_wx[valid_fits] * S_wy[valid_fits])) / delta[valid_fits]
            
            # 5. Vectorized Anisotropy Inversion
            NZ = np.cos(theta_b)
            denoms = slopes * (1.0 - NZ**2) + NZ
            
            alphas = np.full(NRUNS, np.nan)
            valid_denoms = np.abs(denoms) > 1e-9
            
            alphas[valid_denoms] = (-slopes[valid_denoms] / denoms[valid_denoms]) + 1.0
            
            # 6. Filter for Physical Results (Tani > 0)
            res = alphas[np.isfinite(alphas) & (alphas > 0)]
            
            # Calculate Percentiles
            if len(res) > 10:
                delta_plus[i,j]  = np.percentile(res, 84) - alpha
                delta_minus[i,j] = alpha - np.percentile(res, 16)

            # for k in range(NRUNS):
            #     if len(weights[k][mask[k]]) > 1:
            #         popt = find_slopes(linear_model, xdata[mask[k]], vx_model[k][mask[k]], weights[k][mask[k]])
            #         _, Tani = _calc_anisotropy_from_fits(popt, np.cos(theta_b))

            #         if np.isfinite(Tani):
            #             slopes.append(popt[0])
            #             results.append(Tani)
            #     else:
            #         print(f'bad initialization for k = {k}')

            # if len(results) > 10:
            #     res = np.array(results)
            #     delta_plus[i,j]  = np.percentile(res, 84) - alpha
            #     delta_minus[i,j] = alpha - np.percentile(res, 16)

            """
            # Now, let us use the known information to define the ideal slope!
            mask = normed_vdf > 0.5
            
            xdata = vz_array * np.sin(theta_b)

            weights = normed_vdf
            popt = find_slopes(linear_model, xdata[mask], vx_model[mask], weights[mask])
            alpha, Tani = _calc_anisotropy_from_fits(popt, np.cos(np.radians(theta_b)))

            # Check the density
            DEN = np.trapezoid(F_1D, x=vz_array)
            """

