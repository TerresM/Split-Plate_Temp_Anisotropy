import numpy as np
import matplotlib.pyplot as plt
import cdflib.xarray
import numpy as np

from matplotlib.colors import LogNorm

plt.ion()
from scipy.ndimage import gaussian_filter

def smooth_for_contours(data, sigma=1.5):
    """
    Applies a Gaussian blur to a 2D array to smooth contour lines, 
    while mathematically ignoring NaNs so they don't bleed into the data.
    """
    # Numerator: Replace NaNs with 0s and smooth
    V = data.copy()
    V[np.isnan(data)] = 0
    VV = gaussian_filter(V, sigma=sigma)
    
    # Denominator: Weight matrix (1 for valid data, 0 for NaN) and smooth
    W = np.ones_like(data)
    W[np.isnan(data)] = 0
    WW = gaussian_filter(W, sigma=sigma)
    
    # Normalize the smoothed data
    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = np.where(WW != 0, VV / WW, np.nan)
    
    # Re-apply the exact original NaN mask to keep the hatched borders sharp
    smoothed[np.isnan(data)] = np.nan
    
    return smoothed


data_map = cdflib.xarray.cdf_to_xarray('./FMCMC_uncert_map_final_with_logvalues_count_and_bias.cdf')

data_map = data_map.where(data_map != -1e31, np.nan)

# Get the counts
counts = data_map['count'].data
max_counts = np.max(counts)

# We need to drop bins that do not have a measureable max current. 
mask = (counts < 0.7*max_counts)

# Load in the data maps. 
delta_plus  = data_map.delta_plus.data
delta_minus = data_map.delta_minus.data

# Get the upper and lower bound uncertainty maps.
delta_plus[mask]  = np.nan #np.nanmax(delta_plus)
delta_minus[mask] = np.nan #np.nanmax(delta_minus)

bias = data_map.bias.data

bias[mask] = np.nan

# Now we will make our figure. 
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

# Change to 3 rows, and increase the height from 10 to 14
fig, ax = plt.subplots(3, 1, figsize=(8, 16), layout='constrained', sharex=True, sharey=True)

A, T = np.meshgrid(data_map.anisotropy.data, data_map.theta.data, indexing='ij')

# Calculate Percentages
perc_upper = (10**(delta_plus) - 1.0) * 100
perc_lower = (1.0 - 10**(-delta_minus)) * 100

linear_offset = bias

# Define label positions to keep contours clean
label_positions = [(30, 2.0), (50, 0.3)] 

# ---------------------------------------------------------
# Panel (a): Delta Plus (Top)
# ---------------------------------------------------------
im1 = ax[0].pcolormesh(T, A, perc_upper, cmap='inferno', norm=LogNorm(vmin=5, vmax=100))
cb1 = fig.colorbar(im1, ax=ax[0])
cb1.set_label(r'Upper Error $\Delta^+$ (%)', fontsize=14)
ax[0].set_title(r'(a) Upper Uncertainty Map', loc='left', fontsize=16)
ax[0].set_ylabel(r'Anisotropy $T_{\perp}/T_{\parallel}$')

# Add smoothed contours to Panel (a)
cs1 = ax[0].contour(T, A, perc_upper, levels=[20, 50], colors=['white','black'], linewidths=2, linestyles=['--', '--'])
ax[0].clabel(cs1, inline=True, fontsize=12, fmt='%1.0f%%', use_clabeltext=True)#, manual=label_positions)

# ---------------------------------------------------------
# Panel (b): Delta Minus (Middle)
# ---------------------------------------------------------
im2 = ax[1].pcolormesh(T, A, perc_lower, cmap='inferno', norm=LogNorm(vmin=5, vmax=100))
cb2 = fig.colorbar(im2, ax=ax[1])
cb2.set_label(r'Lower Error $\Delta^-$ (%)', fontsize=14)
ax[1].set_title(r'(b) Lower Uncertainty Map', loc='left', fontsize=16)
ax[1].set_ylabel(r'Anisotropy $T_{\perp}/T_{\parallel}$')

# Add smoothed contours to Panel (b)
cs2 = ax[1].contour(T, A, perc_lower, levels=[20, 50], colors=['white', 'black'], linewidths=2, linestyles=['--', '--'])
ax[1].clabel(cs2, inline=True, fontsize=12, fmt='%1.0f%%', use_clabeltext=True)#, manual=label_positions)

# ---------------------------------------------------------
# Panel (c): Bias (Bottom)
# ---------------------------------------------------------
im3 = ax[2].pcolormesh(T, A, linear_offset, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
cb3 = fig.colorbar(im3, ax=ax[2])
cb3.set_label(r'Bias ($\log_{10}$)', fontsize=14)
ax[2].set_title(r'(c) Bias Map', loc='left', fontsize=16)
ax[2].set_ylabel(r'Anisotropy $T_{\perp}/T_{\parallel}$')

# ---------------------------------------------------------
# Global Axis Labels and Scaling
# ---------------------------------------------------------
# Because of sharex=True, we only put the X-label on the bottom plot
ax[2].set_xlabel(r'Magnetic Flow Angle $\theta_{B}$ $[^{\circ}]$')

for axis in ax:
    axis.set_yscale('log')
    
    # Gray out the NaN regions
    axis.set_facecolor('lightgray') 
    
    # Add a cross-hatch pattern to the NaN regions
    axis.patch.set_hatch('///') 
    
    # Make the hatching a bit softer (dark gray instead of pure black)
    axis.patch.set_edgecolor('dimgray')

plt.savefig('./Updated_uncertainty_map_3panel.png', dpi=300, bbox_inches='tight')
plt.show()