import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cdflib.xarray 
import xarray as xr
import sys

import matplotlib.dates as mdates

from functions.span_flow_angle_calc import flow_angles_span_inst
plt.ion()

def Partial_T_anisotropy(Txx, Tzz, b_vec):
    # This function calculates the temperature anisotropy using Badman_et_al_2025
    '''
    Parameters:
    -----------
    t_tens : ndarray
        The symmetric temperature tensor. 
    b_vec : ndarray
        The magnetic field vector. 
    Outputs:
    --------
    t_fa : ndarray
        Temperature tensor in field aligned coordinates
    '''
    magb = np.linalg.norm(b_vec, axis=1)
    theta = np.arccos(b_vec[:,2]/magb)

    T_perp = Tzz + (Txx - Tzz)/(1 - ((b_vec[:,0]/magb)**2) * np.tan(theta)**2)
    T_para = T_perp + (Txx - Tzz)/(((b_vec[:,0]/magb)**2) * np.sin(theta)**2 - np.cos(theta)**2)

    T_fa = np.vstack([T_perp, T_para])

    return(T_fa)

file_str = '2023-12-27'

year, month, day = file_str.split('-') 
bimax = cdflib.xarray.cdf_to_xarray(f'./v00/{year}/{month}/spp_swp_spi_sf00_fits_{file_str}_v00.cdf', fillval_to_nan=True)
spc   = cdflib.xarray.cdf_to_xarray(f'/home/michael/Research/PSP_Temp_Anisotropy/anisotropy_files/Bayesian_MCMC_5D_NCP/SPC_Anisotropy_{file_str}_peak_track_Cauchy_fit4s_gflag.cdf', fillval_to_nan=True)
span  = cdflib.xarray.cdf_to_xarray(f'./psp_data/sweap/spi/L3/spi_sf00/{year}/{month}/psp_swp_spi_sf00_L3_mom_{year}{month}{day}_v04.cdf')

spc = spc.rename({'Epoch':'time'})
span = span.drop('ROTMAT_SC_INST')

# Set all fill_vals to be nans.
bimax = bimax.where(bimax != -1e31, np.nan)
spc = spc.where(spc != -1e31, np.nan)
span = span.where(span != -1e31, np.nan)

# Get key values that we are going to be plotting. 
spc_time = spc.time.data

spc_ani = spc.T_ani_avg.data
spc_ani_upper = spc_ani + np.sqrt(spc.T_ani_avg_delta_plus.data)
spc_ani_lower = spc_ani - np.sqrt(spc.T_ani_avg_delta_minus.data)

chi_ani = spc.T_ani_chi.data
chi_ani_upper = chi_ani + spc.T_ani_chi_delta_plus.data
chi_ani_lower = chi_ani - spc.T_ani_chi_delta_minus.data

psi_ani = spc.T_ani_psi.data
psi_ani_upper = psi_ani + spc.T_ani_psi_delta_plus.data
psi_ani_lower = psi_ani - spc.T_ani_psi_delta_minus.data

# Interpolate the SPAN data onto the SPC time resolution.
span_interp = span.interp(Epoch=spc.time.data)

# From the bimax files we will alawys assume the core has a higher density.
den1 = bimax.np1.values
den2 = bimax.np2.values

Trat1 = bimax.Trat1
Trat2 = bimax.Trat2

T_bimax_fits = np.zeros(len(Trat1))

T_bimax_fits[den1 > den2] = Trat1[den1 > den2]
T_bimax_fits[den2 > den1] = Trat2[den2 > den1]  
# T_bimax_fits = Trat1

bimax['Temp_ani'] = xr.DataArray(T_bimax_fits, coords={'Epoch':bimax.Epoch.data})

bimax_interp = bimax.interp(Epoch=spc.time.data)

# Partial moment span-i temperature anisotropy.
# Tperp_span, Tpara_span = Partial_T_anisotropy(span_interp.T_TENSOR_INST.data[:,0], span_interp.T_TENSOR_INST.data[:,2], span_interp.MAGF_INST.data)
# Tani_span = Tperp_span/Tpara_span
T_fa = Partial_T_anisotropy(span_interp.T_TENSOR_INST.data[:,0], span_interp.T_TENSOR_INST.data[:,2], span_interp.MAGF_INST.data)
Tperp_span, Tpara_span = T_fa

Tani_span = Tperp_span/Tpara_span

# Get the span-i flow angles
el, az = flow_angles_span_inst(span_interp.VEL_INST.data)

# choose only the points that have two fits together. 
chi_flag = np.isfinite(spc.T_ani_chi.data)
psi_flag = np.isfinite(spc.T_ani_psi.data)

two_fits_flag = chi_flag & psi_flag

# Get the degree threshold.
deg_thres = 2.0
az_flag = np.abs(spc.peak_azimuth - az) < deg_thres
el_flag = np.abs(spc.peak_elevation - el) < deg_thres

angle_flag = az_flag & el_flag

# Let us downsample the temperature anisotropy to 1-minute.
spc_mean_interp = spc.rolling(time=30, center=True, min_periods = 1).mean()#spc.resample(time='1min').mean()
span_mean_interp = span_interp.rolling(Epoch=30, center=True, min_periods=1).mean() #span_interp.resample(Epoch='1min').mean()
bimax_mean_interp = bimax_interp.rolling(Epoch=30, center=True, min_periods=1).mean() #bimax_interp.resample(Epoch='1min').mean()

T_fa_mean = Partial_T_anisotropy(span_mean_interp.T_TENSOR_INST.data[:,0], span_mean_interp.T_TENSOR_INST.data[:,2], span_mean_interp.MAGF_INST.data)
Tperp_span_mean, Tpara_span_mean = T_fa_mean

Tani_span_mean = Tperp_span_mean/Tpara_span_mean

#### Begin our figure
fig = plt.figure(figsize=(12, 10), layout='constrained')

# Slightly reduced the top panel ratio just to give the angle panels breathing room
gs = fig.add_gridspec(6, 1, height_ratios=[2.5, 1, 1, 1, 1, 1], hspace=0.05)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[3], sharex=ax1)
ax5 = fig.add_subplot(gs[4], sharex=ax1)
ax6 = fig.add_subplot(gs[5], sharex=ax1)

# Only the Anisotropy panel should be log scale!
ax1.set_yscale('log')
ax1.set_ylim([0.1, 10])

# Custom Colors
# color1 = "#067FCF" # Teal (Partial Moments)
color1 = "#0555EA"
color2 = "#C87100FF" # Orange (Bi-Max Fits)
color3 = "#8504D5FF"

ax1.plot(spc_time, spc_ani, color='k', linewidth=0.5, zorder=0, label='SPC Anisotropy')
ax1.fill_between(spc_time, spc_ani_lower, spc_ani_upper, 
                 color='tab:red', alpha=0.5, edgecolor='none', zorder=0, label='SPC Uncertainty')

ax1.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

ax1.scatter(span_interp.Epoch.data[angle_flag], Tani_span[angle_flag],
            marker='o', color=color1, edgecolors='k', linewidths=0.4, s=10, alpha=0.4, zorder=2, label='SPAN-i Partial')

# ax1.scatter(spc_time[angle_flag], bimax_interp.Temp_ani.data[angle_flag], 
#             marker='v', color=color3, edgecolors='k', linewidths=0.4, s=15, alpha=0.4, zorder=2, label='SPAN-i Bi-Max')

ax1.set_xlim([spc_time[0] + np.timedelta64(1, 'h'), spc_time[0] + np.timedelta64(6, 'h')])

# Add a clean legend inside the top panel
leg1 = ax1.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray', ncol=2)

# for handle in leg1.legend_handles:
#     handle.set_linewidth(3.0)

# Add faint gridlines to help the eye track across the log scale
ax1.grid(True, which='major', color='gray', alpha=0.3, linestyle='-')
ax1.grid(True, which='minor', color='gray', alpha=0.1, linestyle='-')

# Add in the magnetic field unit vector.
# ax2.plot(spc_time, spc.Nx, linewidth=0.5)
# ax2.plot(spc_time, spc.Ny, linewidth=0.5)
# ax2.plot(spc_time, spc.Nz, linewidth=0.5)
ax2.plot(spc_time, spc.B_x_flow_angle, linewidth=0.5, color='tab:green', label=r'$\phi_{B}$')
ax2.plot(spc_time, spc.B_y_flow_angle, linewidth=0.5, color='tab:orange', label=r'$\theta_{B}$')
ax2.axhline(0, linestyle='dashed', color='k', linewidth=2)
leg2 = ax2.legend(ncol=2, framealpha=0.9, edgecolor='lightgray')

for handle in leg2.legend_handles:
    handle.set_linewidth(3.0)

# Add in the radial velocity
ax3.plot(spc_time, spc.vp_fit_SC.data[:,2], color='k', linewidth=0.5, label='SPC Fit')
ax3.plot(span.Epoch.data, span.VEL_SC.data[:,2], color=color1, alpha=0.5, linewidth=0.5, label='SPAN Moment')
leg3 = ax3.legend(ncol=3, framealpha=0.9, edgecolor='lightgray')

for handle in leg3.legend_handles:
    handle.set_linewidth(3.0)

# Add in the flow angle plots
ax4.plot(spc_time, spc.peak_elevation.data, color='k', linewidth=0.5)
ax4.plot(spc_time[0:6778], el[0:6778], color=color1, linewidth=0.5, alpha=0.5)

ax5.plot(spc_time, spc.peak_azimuth.data, color='k', linewidth=0.5)
ax5.plot(spc_time[0:6778], az[0:6778], color=color1, linewidth=0.5, alpha=0.5)
ax5.axhline(-5.625, color=color3, linestyle='dashed')

# 1. Extract the raw arrays
time_data = span.Epoch.data
phi_data = span.PHI_VALS.data - 180  # Update this if your phi variable is named differently
eflux_data = span.EFLUX_VS_PHI.data

# 2. Broadcast the 1D time array to match the 2D shape of phi and eflux
# time_data[:, None] turns shape (T,) into (T, 1)
# np.broadcast_to expands it to (T, P) without duplicating memory
time_2d = np.broadcast_to(time_data[:, None], eflux_data.shape)

# 3. Create the 2D plot
# We use LogNorm because space plasma fluxes usually span multiple orders of magnitude.
# shading='nearest' or 'auto' keeps the grid clean when working with dynamic 2D coordinates.
im = ax6.pcolormesh(
    time_2d, 
    phi_data, 
    eflux_data, 
    cmap='Spectral_r',    # 'Spectral_r', 'viridis', or 'plasma' are standard for flux
    norm=colors.LogNorm(), # Automatically scales the colormap logarithmically
    shading='auto'
)

# 4. Attach a colorbar to the specific axis
cbar = plt.colorbar(im, ax=ax6, pad=0.02)
cbar.set_label('Energy Flux')

# 5. Format the axis labels
ax6.set_ylabel('Phi Angle [deg]')

# Clean up axes: Hide X-tick labels for the top 3 panels
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)
ax3.tick_params(labelbottom=False)
ax4.tick_params(labelbottom=False)
ax5.tick_params(labelbottom=False)

# Optional: Add y-labels just to be safe
ax1.set_ylabel(r'$T_{\perp} / T_{\parallel}$', fontsize=16)
ax2.set_ylabel(r'($\phi_{B},\theta_{B}$) [$^\circ$]', fontsize=16)
ax3.set_ylabel(r'$v_{r}$ [km/s]', fontsize=16)
ax4.set_ylabel(r'$\theta$ [$^\circ$]', fontsize=16)
ax5.set_ylabel(r'$\phi$ [$^\circ$]', fontsize=16)

# Add faint gridlines to help the eye track across the log scale
ax2.grid(True, which='major', color='gray', alpha=0.3, linestyle='-')
ax2.grid(True, which='minor', color='gray', alpha=0.1, linestyle='-')

ax3.grid(True, which='major', color='gray', alpha=0.3, linestyle='-')
ax3.grid(True, which='minor', color='gray', alpha=0.1, linestyle='-')

ax4.grid(True, which='major', color='gray', alpha=0.3, linestyle='-')
ax4.grid(True, which='minor', color='gray', alpha=0.1, linestyle='-')


ax1.set_title(f'Temperature Anisotropy Comparision | {file_str}', fontsize=20)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax6.set_xlabel(r'Time [UTC] on 2023-12-27', fontsize=12)

axes = [ax1, ax2, ax3, ax4, ax5, ax6]
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for ax, label in zip(axes, labels):
    # Place the text at x=0.01 (1% from the left) and y=0.95 (5% from the top)
    ax.text(0.01, 0.92, label, 
            transform=ax.transAxes,   # Use axes coordinates, not data coordinates
            fontsize=14, 
            fontweight='bold', 
            va='top',                 # Align the text vertically to the top
            ha='left',                # Align the text horizontally to the left
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), # Optional
            zorder=10)                # Ensure it sits on top of all data
fig.align_ylabels()

# plt.savefig('./figures/figure_Anisotropy_comparision.png')