import numpy as np
import xarray as xr
import cdflib.xarray
import matplotlib.pyplot as plt
import multiprocessing as mp

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
    dvz_array = np.abs(np.gradient(vz_array))
    conversion_factor = A_eff * q_e * np.abs(vz_array) * dvz_array * C_cal

    I_ideal = F_1D * conversion_factor  # This is a proxy for the current.

    # We assume there is a 1 pA noise floor
    I_noise_bg   = np.random.normal(0, 1, size = len(vz_array))

    I_model = I_ideal + I_noise_bg + 1.0

    # Define the reduced vdf model
    F_1D_model = I_model/ conversion_factor
    F_1D_model = np.clip(F_1D_model, 0, None)

    return(F_1D_model, I_ideal)

def build_1d_vdf_clean_vectorized(VX, VZ, n_arr, ux_arr, uz_arr, wp_arr, theta_bn, R_true, FOV_mask, R_alpha, vx_array):
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
    f_2d_masked = f_2d * R_alpha[None, :, :] #FOV_mask[None, :, :]

    # Integrate across the vx_array (which is axis 1 in our 3D setup)
    F_1D_all = np.trapezoid(f_2d_masked, x=vx_array, axis=1)

    return F_1D_all  # Shape is perfectly (NRUNS, Nz)

def build_1d_vdf_model_vectoriezed(f_1ds, vz_array):
    A_eff = 1.59
    q_e   = 1.602e-19
    C_cal = 1e5 * 1e12 
    dvz_array = np.abs(np.gradient(vz_array))
    conversion_factor = A_eff * q_e * np.abs(vz_array) * dvz_array * C_cal

    I_ideal = f_1ds * conversion_factor  # This is a proxy for the current.

    I_noise_bg = np.random.normal(0, 1, size=I_ideal.shape)

    I_model = I_ideal + I_noise_bg + 1.0

    # Define the reduced vdf model
    f_1ds_models = I_model/conversion_factor
    f_1ds_models = np.clip(f_1ds_models, 0, None)

    return f_1ds_models, I_ideal

def process_grid_point(args):
    """
    Standalone worker function to process a single (alpha, theta) grid point.
    Unpacks arguments, generates the Monte Carlo VDFs, runs Vectorized WLS, 
    and returns the calculated percentiles.
    """
    # Unpack the arguments passed from the multiprocessing pool
    i, alpha, j, theta_b, n_arr, ux_arr, uz_arr, wp_arr, VX, VZ, FOV_mask, R_alpha, vx_array, vz_array, NRUNS = args
    
    # 1. Generate Clean VDFs
    f_1ds = build_1d_vdf_clean_vectorized(VX, VZ, n_arr, ux_arr, uz_arr, wp_arr, theta_b, alpha, FOV_mask, R_alpha, vx_array)
    
    # 2. Apply Noise Model
    f_1ds_models, I_ideal = build_1d_vdf_model_vectoriezed(f_1ds, vz_array)
    
    # 3. Calculate Ideal Velocity and Transverse Noise
    m_ideal  = ((1 - alpha) * np.cos(theta_b)) / (np.cos(theta_b)**2 + alpha * np.sin(theta_b)**2)
    vx_ideal = m_ideal * vz_array * np.sin(theta_b)
    
    # sigma_vx = 5.0 * (1.0 / np.sqrt(I_ideal + 0.01))
    # sigma_vx = 100.0 * (1.0 / (I_ideal + 0.01))
    # vx_noise = np.random.normal(0, 1, size=I_ideal.shape) * sigma_vx
    # vx_model = vx_ideal + vx_noise
    I_norm = I_ideal / (np.max(I_ideal) + 1e-12)

    # popt = [ -4.08906605,  35.15915399, -32.52587594]
    popt = [-0.00118529,  0.01581001, -0.01449879]
    sigma_theta_deg = (popt[0]/(I_norm)) + (popt[1]/np.sqrt(I_norm)) + popt[2]
    sigma_theta_rad = np.radians(sigma_theta_deg)

    sigma_vx = vz_array * sigma_theta_rad
    vx_noise = np.random.normal(0, 1, size=I_ideal.shape) * sigma_vx

    vx_model = vx_ideal + vx_noise

    # 4. Prepare Weights for Fitter
    xdata = vz_array * np.sin(theta_b)
    
    # Add a tiny epsilon (1e-12) to prevent divide-by-zero warnings if a VDF is totally wiped out
    max_f1ds = np.max(f_1ds_models, axis=1)[:, None] + 1e-12 
    weights = f_1ds_models / max_f1ds
    mask = weights > 0.5

    # 5. Fast Vectorized Least Squares
    X = xdata[None, :]
    Y = vx_model
    
    W = np.zeros_like(Y)
    W[mask] = 1.0 / (weights[mask]**2)
    
    S_w   = np.sum(W, axis=1)
    S_wx  = np.sum(W * X, axis=1)
    S_wy  = np.sum(W * Y, axis=1)
    S_wxx = np.sum(W * X * X, axis=1)
    S_wxy = np.sum(W * X * Y, axis=1)
    
    delta = (S_w * S_wxx) - (S_wx**2)
    
    valid_fits = (np.sum(mask, axis=1) > 1) & (np.abs(delta) > 1e-12)
    
    slopes = np.full(NRUNS, np.nan)
    slopes[valid_fits] = ((S_w[valid_fits] * S_wxy[valid_fits]) - 
                          (S_wx[valid_fits] * S_wy[valid_fits])) / delta[valid_fits]
    
    # 6. Anisotropy Inversion
    NZ = np.cos(theta_b)
    denoms = slopes * (1.0 - NZ**2) + NZ
    
    alphas = np.full(NRUNS, np.nan)
    valid_denoms = np.abs(denoms) > 1e-9
    alphas[valid_denoms] = (-slopes[valid_denoms] / denoms[valid_denoms]) + 1.0
    
    res = alphas[np.isfinite(alphas) & (alphas > 0)]
    
    # 7. Calculate Percentiles
    d_plus, d_minus = np.nan, np.nan
    count = len(res)
    if len(res) >= 0.5 * NRUNS:
        d_plus  = np.percentile(res, 84) - alpha
        d_minus = alpha - np.percentile(res, 16)
        # d_plus = np.max(res) - alpha
        # d_minus = alpha - np.min(res)
        
    return (i, j, d_plus, d_minus, count)

if __name__ == "__main__":
    # Load in the samples, let us start with the moments.
    samples = cdflib.xarray.cdf_to_xarray('spc_all_valid_moms.cdf')

    sample_mask = (samples.np_moment.data > 10) & (samples.wp_moment.data > 20) & (samples.np_moment.data < 2000)

    samples = samples.isel(Epoch=sample_mask)

    # High resolution grid, this is to be comparable to peak tracks
    speed_bins = np.logspace(np.log10(138), np.log10(875), num=128) # Assume 60 energy windows
    vz_array = -np.sort(speed_bins)

    vx_limit = np.max(np.abs(vz_array)) * np.tan(np.radians(30))
    vx_array = np.linspace(-vx_limit, vx_limit, 60)

    VX, VZ = np.meshgrid(vx_array, vz_array, indexing='ij')

    incident_angles = np.abs(np.arctan2(VX, np.abs(VZ)))
    FOV_mask = incident_angles <= np.radians(30.0)    

    # Updating the effective area calculation
    A_0 = 131.783   # mm^2, slope = -1.8 mm^2/deg
    A_alpha = A_0 - (np.degrees(incident_angles) * 1.8)

    steep_mask = incident_angles > 32.0
    if np.any(steep_mask):
        A_alpha[steep_mask] = (A_0 - (32.0 * 1.8)) - ((incident_angles[steep_mask] - 32.0) * 6.9)
    
    A_alpha = np.clip(A_alpha, 0, None)
    R_alpha = A_alpha/A_0       # This is effectively the transparancy factor or the instruments affective area. 


    NTH, NANI = 50, 50 # Let's max it out!
    theta_vals = np.linspace(0.3, 89.5, NTH)
    ani_vals = np.logspace(np.log10(0.05), np.log10(15.0), NANI)

    delta_plus = np.full((NANI, NTH), np.nan)
    delta_minus = np.full((NANI, NTH), np.nan)
    # NRUNS = 6000
    NRUNS = 600

    # 1. Sample the plasma states ONCE for the entire grid
    print("Sampling base plasma states...")
    idxs = np.random.randint(0, len(samples.Epoch.data), size=(NRUNS))
    n_arr = samples.np_moment.values[idxs]
    ux_arr = samples.vp_moment_SC.values[idxs,0]
    uz_arr = samples.vp_moment_SC.values[idxs,2]
    wp_arr = samples.wp_moment.values[idxs]

    # 2. Build the Task List
    print("Building task list...")
    tasks = []
    for i in range(NANI):
        alpha = ani_vals[i]
        for j in range(NTH):
            theta_b = np.radians(theta_vals[j])
            
            # Pack all necessary variables into a single tuple
            task_args = (i, alpha, j, theta_b, n_arr, ux_arr, uz_arr, wp_arr, VX, VZ, FOV_mask, R_alpha, vx_array, vz_array, NRUNS)
            tasks.append(task_args)

    # 3. Ignite the CPU Pool
    num_cores = 8 #mp.cpu_count()
    print(f"Firing up multiprocessing pool with {num_cores} cores...")
    
    count = np.zeros(delta_minus.shape)
    # Pool automatically handles chunking and load balancing
    with mp.Pool(processes=num_cores) as pool:
        # imap_unordered yields results as soon as any core finishes a point
        for result in tqdm(pool.imap_unordered(process_grid_point, tasks), total=len(tasks)):
            i, j, d_plus, d_minus, counts = result
            
            # Store the computed bounds into your matrices
            delta_plus[i, j] = d_plus
            delta_minus[i, j] = d_minus
            count[i, j] = counts
            
    print("Grid evaluation complete! Ready to plot.")

    dp_xr = xr.DataArray(delta_plus, coords = {'anisotropy':ani_vals, 'theta':theta_vals})
    dm_xr = xr.DataArray(delta_minus, coords = {'anisotropy':ani_vals, 'theta':theta_vals})

    xr_ds = xr.Dataset({'delta_plus':dp_xr, 'delta_minus':dm_xr})

    A, T = np.meshgrid(ani_vals, theta_vals, indexing='ij')

    fig, ax = plt.subplots(2, 2, figsize=(12,8), layout='constrained', sharey=True, sharex=True)

    a1 = ax[0, 0].pcolormesh(T, A, delta_plus, norm=LogNorm())
    a2 = ax[0, 1].pcolormesh(T, A, delta_minus, norm=LogNorm())
    a3 = ax[1, 0].pcolormesh(T, A, (100*delta_plus)/A , norm=LogNorm(vmax=100))
    a4 = ax[1, 1].pcolormesh(T, A, (100*delta_minus)/A, norm=LogNorm(vmax=100))

    [ax[i,j].set_yscale('log') for i in range(2) for j in range(2)]

    [ax[i, j].set_ylabel(r'$T_{\perp}/T_{\parallel}$', fontsize=18) for i in range(2) for j in range(2)]
    [ax[i, j].set_xlabel(r'$\theta_{B} [^{\circ}]$', fontsize=18) for i in range(2) for j in range(2)]

    titles = [r'$\Delta^{+}$', r'$\Delta^{-}$', r'% Uncertainty $\Delta^{+}$', r'% Uncertainty $\Delta^{-}$']

    [ax[i, j].set_title(titles[2*i+j], fontsize=18) for i in range(2) for j in range(2)]

    fig.colorbar(a1)
    fig.colorbar(a2)
    fig.colorbar(a3)
    fig.colorbar(a4)


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

