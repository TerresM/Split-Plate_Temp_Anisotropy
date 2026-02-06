import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

def slope_model(R, TH):
    TH_rad = np.radians(TH)

    Nx = np.sin(TH_rad)
    Nz = np.cos(TH_rad)

    slope = ((1 - R) * Nz)/(1 - (1 - R)*(1 - Nx**2))

    return(slope)

def alpha_model(M, TH):
    TH_rad = np.radians(TH)

    Nx = np.sin(TH_rad)
    Nz = np.cos(TH_rad)

    alpha = 1 - (M / (M * (1 - Nz**2) + Nz))

    return(alpha)



if __name__ == "__main__":
    aniso_vals = np.logspace(np.log10(0.03), np.log10(30), 51)
    theta_vals = np.linspace((0.1), (89.9), 101)

    AN, TH = np.meshgrid(aniso_vals, theta_vals, indexing='ij')

    slope = slope_model(AN, TH)
    alpha = alpha_model(slope, TH)

    fig, ax = plt.subplots(1, 2, figsize=(12,6), layout='constrained')
    a0 = ax[0].pcolormesh(theta_vals, aniso_vals, slope, cmap='inferno', norm=SymLogNorm(linthresh=0.1))
    a1 = ax[1].pcolormesh(theta_vals, aniso_vals, np.sqrt(alpha)**2, cmap='inferno', norm=LogNorm())
    [ax[i].set_yscale('log') for i in range(2)]
    plt.colorbar(a0)
    plt.colorbar(a1)
    plt.show()