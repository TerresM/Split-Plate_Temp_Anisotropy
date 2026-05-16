import numpy as np
import matplotlib.pyplot as plt
import cdflib.xarray
import numpy as np

from matplotlib.colors import LogNorm

data_map = cdflib.xarray.cdf_to_xarray('/home/michael/Research/PSP_Temp_Anisotropy/FMCMC_uncert_map_final_with_count_v7.cdf')

data_map = data_map.where(data_map != -1e31, np.nan)

# Get the counts
counts = data_map['count'].data
max_counts = np.max(counts)

# We need to drop bins that do not have a measureable max current. 
mask = (counts < 0.9*max_counts)

# Load in the data maps. 
delta_plus  = data_map.delta_plus.data
delta_minus = data_map.delta_minus.data

# Get the upper and lower bound uncertainty maps.
delta_plus[mask]  = np.nanmax(delta_plus)
delta_minus[mask] = np.nanmax(delta_minus)

delta_plus[np.isnan(delta_plus)]   = np.nanmax(delta_plus)
delta_minus[np.isnan(delta_minus)] = np.nanmax(delta_minus)

# # Now we will make our figure. 
# fig, ax = plt.subplots(2, figsize=(8, 12), layout='constrained', sharex=True)

# A, B = np.meshgrid(data_map.anisotropy.data, data_map.theta.data, indexing='ij')

# im = ax[0].pcolormesh(B, A, 100*delta_plus/A, cmap='inferno', norm=LogNorm(vmin=1, vmax=100))
# im2 = ax[1].pcolormesh(B, A, 100*delta_minus/A, cmap='inferno', norm=LogNorm(vmin=1, vmax=100))

# titles = ['$\Delta^+$ % Uncertainty', '$\Delta^-$ % Uncertainty']
# for i in range(2):
#     ax[i].set_yscale('log')
#     ax[i].tick_params(axis='both', labelsize=14)
#     ax[i].set_title(titles[i], fontsize=18)

# ax[0].set_ylabel(r'$T_{\perp}/T_{\parallel}$', fontsize=18)
# ax[1].set_ylabel(r'$T_{\perp}/T_{\parallel}$', fontsize=18)

# ax[1].set_xlabel(r'$\Theta_{B} [^{\circ}]$', fontsize=18)

# cbar = fig.colorbar(im, ax=ax, location='right')
# cbar.set_label('Uncertainty (%)', fontsize=18)
# cbar.ax.tick_params(labelsize=14)

# plt.savefig('./figures/uncertainty_map.png')
# plt.show()

# Now we will make our figure. 
fig, ax = plt.subplots(2, figsize=(8, 12), layout='constrained', sharex=True)

A, B = np.meshgrid(data_map.anisotropy.data, data_map.theta.data, indexing='ij')

# Pre-calculate the percent uncertainties to keep the code clean
uncert_plus = 100 * delta_plus / A
uncert_minus = 100 * delta_minus / A

im = ax[0].pcolormesh(B, A, uncert_plus, cmap='inferno', norm=LogNorm(vmin=1, vmax=100))
im2 = ax[1].pcolormesh(B, A, uncert_minus, cmap='inferno', norm=LogNorm(vmin=1, vmax=100))

# --- Add the 20% Uncertainty Contours ---
# We use black dashed lines so they pop beautifully against the 'inferno' colormap
CS0 = ax[0].contour(B, A, uncert_plus, levels=[50], colors='black', linestyles='dashed', linewidths=2)
ax[0].clabel(CS0, inline=True, fontsize=12, fmt='50%%')

CS01 = ax[0].contour(B, A, uncert_plus, levels=[20], colors='black', linestyles='dotted', linewidths=2)
ax[0].clabel(CS01, inline=True, fontsize=12, fmt='20%%')


CS1 = ax[1].contour(B, A, uncert_minus, levels=[50], colors='black', linestyles='dashed', linewidths=2)
ax[1].clabel(CS1, inline=True, fontsize=12, fmt='50%%')

CS11 = ax[1].contour(B, A, uncert_plus, levels=[20], colors='black', linestyles='dotted', linewidths=2)
ax[1].clabel(CS11, inline=True, fontsize=12, fmt='20%%')

# ----------------------------------------

titles = [r'$\Delta^+$ % Uncertainty', r'$\Delta^-$ % Uncertainty']
for i in range(2):
    ax[i].set_yscale('log')
    ax[i].tick_params(axis='both', labelsize=14)
    ax[i].set_title(titles[i], fontsize=18)

ax[0].set_ylabel(r'$T_{\perp}/T_{\parallel}$', fontsize=18)
ax[1].set_ylabel(r'$T_{\perp}/T_{\parallel}$', fontsize=18)

ax[1].set_xlabel(r'$\Theta_{B} [^{\circ}]$', fontsize=18)

cbar = fig.colorbar(im, ax=ax, location='right')
cbar.set_label('Uncertainty (%)', fontsize=18)
cbar.ax.tick_params(labelsize=14)

plt.savefig('./figures/uncertainty_map.png')
plt.show()