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
    magb = np.linalg.norm(b_vec, axis=1)
    theta = np.arccos(b_vec[:,2]/magb)

    T_perp = Tzz + (Txx - Tzz)/(1 - ((b_vec[:,0]/magb)**2) * np.tan(theta)**2)
    T_para = T_perp + (Txx - Tzz)/(((b_vec[:,0]/magb)**2) * np.sin(theta)**2 - np.cos(theta)**2)

    T_fa = np.vstack([T_perp, T_para])
    return(T_fa)

file_str = '2023-12-27'
# file_str = '2025-03-20'
year, month, day = file_str.split('-') 

# Load Data
bimax = cdflib.xarray.cdf_to_xarray(f'./v00/{year}/{month}/spp_swp_spi_sf00_fits_{file_str}_v00.cdf', fillval_to_nan=True)
# spc   = cdflib.xarray.cdf_to_xarray(f'/home/michael/Research/PSP_Temp_Anisotropy/anisotropy_files/Bayesian_MCMC_5D_NCP/SPC_Anisotropy_{file_str}_peak_track_T_fit4s_mf37_gflag.cdf', fillval_to_nan=True)
spc   = cdflib.xarray.cdf_to_xarray(f'/home/michael/Research/PSP_Temp_Anisotropy/anisotropy_files/Bayesian_MCMC_5D_NCP/SPC_Anisotropy_{file_str}_peak_track_T_fit4s_mf38_gflag_v2.cdf', fillval_to_nan=True)
span  = cdflib.xarray.cdf_to_xarray(f'./psp_data/sweap/spi/L3/spi_sf00/{year}/{month}/psp_swp_spi_sf00_L3_mom_{year}{month}{day}_v04.cdf')

l2_support = cdflib.xarray.cdf_to_xarray('/home/michael/Research/PSP_Temp_Anisotropy/span_i_processed_with_fov.cdf')

spc = spc.rename({'Epoch':'time'})
span = span.drop_vars('ROTMAT_SC_INST', errors='ignore')

# Set all fill_vals to be nans.
bimax = bimax.where(bimax != -1e31, np.nan)
spc = spc.where(spc != -1e31, np.nan)
span = span.where(span != -1e31, np.nan)

# Get the b_x and b_y flow angles from the SPC data
bx_angle = spc.B_x_flow_angle.data
by_angle = spc.B_y_flow_angle.data

# SPC Variables
spc_time = spc.time.data
spc_ani = spc.T_ani_avg.data
# Clean up the measurements
pinned_x_flag = (bx_angle < 10) & (spc_ani < 0.2)
pinned_y_flag = (by_angle < 10) & (spc_ani < 0.2)

spc_ani[pinned_x_flag] = np.nan
spc_ani[pinned_y_flag] = np.nan

spc_ani_upper = spc_ani + spc.T_ani_avg_delta_plus.data
spc_ani_lower = spc_ani - spc.T_ani_avg_delta_minus.data

# SPAN Interp & Calculations
span_interp = span.interp(Epoch=spc.time.data)
T_fa = Partial_T_anisotropy(span_interp.T_TENSOR_INST.data[:,0], span_interp.T_TENSOR_INST.data[:,2], span_interp.MAGF_INST.data)
Tperp_span, Tpara_span = T_fa
Tani_span = Tperp_span/Tpara_span

el, az = flow_angles_span_inst(span_interp.VEL_INST.data)

l2_support = l2_support.interp(time=spc.time.data)

# From the bimax files we will alawys assume the core has a higher density.
den1 = bimax.np1.values
den2 = bimax.np2.values

Trat1 = bimax.Trat1
Trat2 = bimax.Trat2

T_bimax_fits = np.zeros(len(Trat1))

T_bimax_fits[den1 > den2] = Trat1[den1 > den2]
T_bimax_fits[den2 > den1] = Trat2[den2 > den1]  

bimax['Temp_ani'] = xr.DataArray(T_bimax_fits, coords={'Epoch':bimax.Epoch.data})

# Angle Flagging
deg_thres = 4.0
az_flag = np.abs(spc.peak_azimuth - az) < deg_thres
el_flag = np.abs(spc.peak_elevation - el) < deg_thres
angle_flag = az_flag & el_flag

# --- NEW SHADING LOGIC ---
# Identify periods where BOTH magnetic flow angles are within +/- 5 degrees
phi_b = spc.B_x_flow_angle.data
theta_b = spc.B_y_flow_angle.data

# Inner core: highly radial, high uncertainty (+/- 10 degrees)
mask_10 = (np.abs(phi_b) < 10.0) & (np.abs(theta_b) < 10.0) & (np.isfinite(spc_ani))

# Outer zone: general uncertainty (+/- 20 degrees)
mask_20 = (np.abs(phi_b) < 15.0) & (np.abs(theta_b) < 15.0) & (np.isfinite(spc_ani))

#### Begin our figure
fig = plt.figure(figsize=(12, 10), layout='constrained')
gs = fig.add_gridspec(6, 1, height_ratios=[2.5, 1, 1, 1, 1, 1], hspace=0.05)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[3], sharex=ax1)
ax5 = fig.add_subplot(gs[4], sharex=ax1)
ax6 = fig.add_subplot(gs[5], sharex=ax1)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

# --- NEW COLOR PALETTE ---
color_spc_line = 'black'
color_spc_unc  = "#e3c37aff" 
color_span     = '#0072B2' 
color_phi      = '#009E73' 
color_theta    = '#D55E00' 

# Apply Shading to all subplots
for ax in axes:
    # Broad, faint shading for general uncertainty (+/- 20 degrees)
    ax.fill_between(spc_time, 0, 1, where=mask_20, 
                    transform=ax.get_xaxis_transform(), 
                    color="#AB5C60", alpha=0.15, zorder=0, lw=0)
    
    # Darker shading for the highly radial core (+/- 10 degrees)
    ax.fill_between(spc_time, 0, 1, where=mask_10, 
                    transform=ax.get_xaxis_transform(), 
                    color="#AB5C60", alpha=0.35, zorder=0, lw=0)

# Panel 1: Anisotropy
ax1.set_yscale('log')
ax1.set_ylim([0.05, 10])

fov_metric = l2_support.fov_metric.data
fov_metric[fov_metric < 0] = np.nan

axx = ax1.twinx()
# axx.set_ylim([0.75,2.0])
# axx.plot(l2_support.time, fov_metric, color='red', linewidth=0.5, alpha=0.2)
# axx.scatter(l2_support.time.data[fov_metric > 0.85], fov_metric[fov_metric > 0.85], color='tab:red', alpha=0.5)
# Shade goes first, then uncertainty, then lines, then scatter
ax1.fill_between(spc_time, spc_ani_lower, spc_ani_upper, 
                 color=color_spc_unc, alpha=1.0, edgecolor='none', zorder=1, label='SPC Uncertainty')
ax1.plot(spc_time, spc_ani, color=color_spc_line, linewidth=0.4, zorder=2, label='SPC Anisotropy')
ax1.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=3)
ax1.scatter(span_interp.Epoch.data[angle_flag], Tani_span[angle_flag],
            marker='o', color=color_span, edgecolors=None, linewidths=None, s=5, alpha=0.5, zorder=4, label='SPAN-i Partial')

# ax1.scatter(bimax.Epoch, bimax.Temp_ani, marker='.', color='red')

leg1 = ax1.legend(loc='lower right', framealpha=0.9, edgecolor='lightgray', ncol=3)
# ax1.set_xlim([spc_time[0] + np.timedelta64(1, 'h'), spc_time[0] + np.timedelta64(6, 'h')])

# Panel 2: Magnetic Angles
ax2.plot(spc_time, phi_b, linewidth=0.8, color=color_phi, label=r'$\phi_{B}$')
ax2.plot(spc_time, theta_b, linewidth=0.8, color=color_theta, label=r'$\theta_{B}$')
ax2.axhline(0, linestyle='dashed', color='black', linewidth=1)
# Add red lines at +/- 5 degrees to visualize the threshold
ax2.axhline(10, linestyle=':', color='tab:red', alpha=0.5, linewidth=1)
ax2.axhline(-10, linestyle=':', color='tab:red', alpha=0.5, linewidth=1)
leg2 = ax2.legend(ncol=2, framealpha=0.9, edgecolor='lightgray', loc='lower right')
for handle in leg2.legend_handles: handle.set_linewidth(3.0)

# Panel 3: Radial Velocity
ax3.plot(spc_time, spc.vp_fit_SC.data[:,2], color=color_spc_line, linewidth=0.8, label='SPC Fit')
ax3.plot(span.Epoch.data, span.VEL_SC.data[:,2], color=color_span, alpha=0.7, linewidth=0.8, label='SPAN Moment')
leg3 = ax3.legend(ncol=3, framealpha=0.9, edgecolor='lightgray', loc='upper right')
for handle in leg3.legend_handles: handle.set_linewidth(3.0)

# Panel 4 & 5: Flow Angles
ax4.plot(spc_time, spc.peak_elevation.data, color=color_spc_line, linewidth=0.8)
ax4.plot(spc_time, el, color=color_span, linewidth=0.8, alpha=0.7)
# ax4.plot(spc_time[0:6778], el[0:6778], color=color_span, linewidth=0.8, alpha=0.7)

ax5.plot(spc_time, spc.peak_azimuth.data, color=color_spc_line, linewidth=0.8)
ax5.plot(spc_time, az, color=color_span, linewidth=0.8, alpha=0.7)
# ax5.plot(spc_time[0:6778], az[0:6778], color=color_span, linewidth=0.8, alpha=0.7)
ax5.axhline(-11.25, color='gray', linestyle='dashed', zorder=1)
ax5.axhline(-32, color='gray', linestyle='dashed', zorder=1)

# Panel 6: Spectrogram
time_data = span.Epoch.data
phi_data = span.PHI_VALS.data - 180  
eflux_data = span.EFLUX_VS_PHI.data
time_2d = np.broadcast_to(time_data[:, None], eflux_data.shape)

# Swapped to 'magma' for a cleaner, modern look
im = ax6.pcolormesh(
    time_2d, phi_data, eflux_data, 
    cmap='magma', 
    norm=colors.LogNorm(vmin=1e9, vmax=1e12), # Added rough vmin/vmax based on your previous image
    shading='auto'
)
cbar = plt.colorbar(im, ax=ax6, pad=0.02)
cbar.set_label('Energy Flux')

# Gridlines, Labels, and Formatting
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)
    ax.grid(True, which='major', color='gray', alpha=0.2, linestyle='-')

ax1.grid(True, which='minor', color='gray', alpha=0.1, linestyle='-')
ax6.set_ylabel('Phi Angle [deg]')

ax1.set_ylabel(r'$T_{\perp} / T_{\parallel}$', fontsize=14)
ax2.set_ylabel(r'($\phi_{B},\theta_{B}$) [$^\circ$]', fontsize=14)
ax3.set_ylabel(r'$v_{r}$ [km/s]', fontsize=14)
ax4.set_ylabel(r'$\theta$ [$^\circ$]', fontsize=14)
ax5.set_ylabel(r'$\phi$ [$^\circ$]', fontsize=14)

ax1.set_title(f'Temperature Anisotropy Comparison | {file_str}', fontsize=18)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax6.set_xlabel(f'Time [UTC] on {file_str}', fontsize=12)

labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for ax, label in zip(axes, labels):
    ax.text(0.01, 0.92, label, transform=ax.transAxes, fontsize=14, fontweight='bold', 
            va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), zorder=10)
fig.align_ylabels()


ax6.set_xlim([np.datetime64(f'{file_str}T01:00:00', 'ns'), np.datetime64(f'{file_str}T06:00:00', 'ns')])
plt.savefig('./figures/figure_Anisotropy_comparision_cleaned.png', dpi=300)
plt.show()