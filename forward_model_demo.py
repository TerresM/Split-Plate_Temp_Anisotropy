import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

# ==========================================
# 1. SETUP PLASMA STATE & GEOMETRY
# ==========================================
# Choose a highly anisotropic "pancake" plasma for a clear visual
R_true = 2.3         # T_perp / T_para
theta_bn = 30.0       # Angle of B-field in degrees
theta_b = np.radians(theta_bn)

n_sample = 50.0       # Density (cm^-3)
uz_sample = 450.0     # Bulk velocity along Z (km/s)
ux_sample = 0.0       # Bulk velocity along X (km/s)
w_sample = 40.0       # Baseline thermal speed (km/s)

# Create the high-resolution velocity grid
vz_array = np.linspace(250, 650, 100)
vx_array = np.linspace(-300, 300, 100)
VZ, VX = np.meshgrid(vz_array, vx_array)  # VZ is X-axis, VX is Y-axis

# ==========================================
# 2. BUILD THE 2D DISTRIBUTION (f_2d)
# ==========================================
bz = np.cos(theta_b)
bx = np.sin(theta_b)

w_para = w_sample / np.sqrt(bz**2 + R_true * (1 - bz**2))
w_perp = np.sqrt(R_true) * w_para

dv_x = VX - ux_sample
dv_z = VZ - uz_sample

v_para = (dv_x * bx) + (dv_z * bz)
v_perp_sq = (dv_z * bx - dv_x * bz)**2

exponent = np.exp(-(v_para**2) / (w_para**2) - (v_perp_sq) / (w_perp**2))
norm = n_sample / (np.pi * w_perp * w_para)
f_2d = norm * exponent

# ==========================================
# 3. APPLY FIELD OF VIEW & INTEGRATE TO 1D
# ==========================================
incident_angles_deg = np.degrees(np.abs(np.arctan2(VX, VZ)))
FOV_mask = incident_angles_deg <= 30.0

# Approximate angular response (A_alpha)
A_0 = 131.783
A_alpha = np.clip(A_0 - incident_angles_deg * 1.8, 0, None)
R_alpha = A_alpha / A_0

f_2d_masked = f_2d * FOV_mask * R_alpha

# Integrate along VX (axis 0 in meshgrid) to get the 1D profile
F_1D = np.trapezoid(f_2d_masked, x=vx_array, axis=0)

# ==========================================
# 4. GENERATE NOISE & FIT THE DATA
# ==========================================
# Convert to a mock normalized current
I_ideal = F_1D * (1.59 * np.abs(vz_array)) 
I_norm = I_ideal / (np.max(I_ideal) + 1e-12)

# Calculate ideal Vx trajectory
# m_ideal = ((1 - R_true) * bz) / (bz**2 + R_true * bx**2)
# vx_ideal = m_ideal * vz_array * bx

# Calculate ideal Vx trajectory
m_ideal = ((1 - R_true) * bz) / (bz**2 + R_true * bx**2)
vx_ideal = ux_sample + m_ideal * (vz_array - uz_sample) * bx

# Apply a mock empirical noise model (similar to what we derived)
sigma_theta_deg = 0.5 / (I_norm + 0.05) + 0.5  # Prevents dividing by zero
sigma_theta_rad = np.radians(sigma_theta_deg)
sigma_vx = vz_array * sigma_theta_rad

# Inject noise to create the "measured" points
vx_noise = np.random.normal(0, 1, size=len(vz_array)) * sigma_vx
vx_measured = vx_ideal + vx_noise

# Fit the linear model
def linear_model(x, a, b): return a * x + b
xdata = vz_array * bx

# Only fit points where the normalized current is > 10% (our threshold)
mask = I_norm > 0.1 

# For curve_fit, sigma represents uncertainty, so we pass 1/weights
weights = I_norm
popt, _ = curve_fit(linear_model, xdata[mask], vx_measured[mask], sigma=1.0/weights[mask])

# Generate the fitted line
vx_fit = linear_model(xdata, *popt)

# ==========================================
# 5. THE MASTER DEMONSTRATION PLOT
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))

# A. The Ideal 2D VDF Background
c = ax.pcolormesh(VZ, VX, f_2d, cmap='viridis', shading='auto', norm=LogNorm(vmin=np.max(f_2d)*1e-1, vmax=np.max(f_2d)))
plt.colorbar(c, ax=ax, label='Phase Space Density')

# Add the dark gray mask over the blind spots (>30 deg)
blind_mask = np.where(incident_angles_deg > 30.0, 1.0, np.nan)
from matplotlib.colors import ListedColormap
ax.pcolormesh(VZ, VX, blind_mask, cmap=ListedColormap(['#202020']), alpha=0.5, zorder=2)

# B. Instrument Line of Sight (Z-axis)
ax.axhline(0, color='white', linestyle=':', linewidth=2, zorder=3, label='Instrument LOS ($Z_{SC}$)')

# C. Magnetic Field Line (Passing through the bulk velocity center)
z_line = np.array([250, 650])
x_line = (z_line - uz_sample) * np.tan(theta_b) + ux_sample
ax.plot(z_line, x_line, color='cyan', linestyle='--', linewidth=2, zorder=3, label=r'Magnetic Field ($\theta_B$)')

# D. The Ideal Vx vs Vz path (Faint Gray)
ax.scatter(vz_array[mask], vx_ideal[mask], color='white', linewidth=1, alpha=0.6, zorder=4, label='Ideal Analytical $V_x$')

# E. The Noisy Data Points (Black Scatter)
ax.scatter(vz_array[mask], vx_measured[mask], color='black', s=60, edgecolor='white', zorder=6, label='Simulated Noisy Measurements')

# F. The Resulting Linear Fit
ax.plot(vz_array[mask], vx_fit[mask], color='red', linewidth=3, zorder=5, label='Weighted Linear Fit')

# Formatting
ax.set_title(f'Forward Model Demonstration ($T_\perp/T_\parallel = {R_true}$, $\\theta_B = {theta_bn}^\circ$)', fontsize=16)
ax.set_xlabel('Velocity $V_z$ (km/s)', fontsize=14)
ax.set_ylabel('Transverse Velocity $V_x$ (km/s)', fontsize=14)
ax.set_xlim(380, 550)
ax.set_ylim(-100, 100)

# Force equal aspect ratio so angles look physically accurate on screen!
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.show()