import os
import sys
import scipy
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d
import cdflib
import cdflib.xarray

def gen_uncertainty_map(angle_bin_width, anisotropy_bins):
    uncertainty_map = cdflib.xarray.cdf_to_xarray(f'{sys.path[0]}/Anisotropy_DELTA_PLUS-MINUS.cdf')

    deltaPlus = uncertainty_map['Anisotropy DELTA_PLUS']
    deltaMinus = uncertainty_map['Anisotropy DELTA_MINUS']

    # now lets define a meshgrid to interpolate the values onto. 
    angleMin = -90
    angleMax = 90
    angleBinWidth = angle_bin_width
    angleBinNumber = int((angleMax - angleMin)/angleBinWidth)

    anisotropyMin = 0.03
    anisotropyMax = 15
    anisotropyBinNumber = anisotropy_bins

    anglelist = np.linspace(angleMin, angleMax, angleBinNumber)
    anisotropylist = np.logspace(np.log10(anisotropyMin), np.log10(anisotropyMax), anisotropyBinNumber)

    # Remove the NaN values
    maskPlus  = (np.any(np.isnan(deltaPlus.data), axis=1))
    maskMinus = (np.any(np.isnan(deltaMinus.data), axis=1))

    # Generate the 2D grid
    X, Y = np.meshgrid(anglelist, anisotropylist, indexing='ij')

    deltaPlus_vals  = scipy.interpolate.griddata((deltaPlus[~maskPlus, 0], deltaPlus[~maskPlus, 1]), deltaPlus[~maskPlus, 2], (X, Y), method='nearest')
    deltaMinus_vals = scipy.interpolate.griddata((deltaMinus[~maskMinus, 0], deltaMinus[~maskMinus, 1]), deltaMinus[~maskMinus, 2], (X, Y), method='nearest')

    return(anglelist, anisotropylist, deltaPlus_vals, deltaMinus_vals)

def gen_uncertainty_map2(angle_bin_width, anisotropy_bins):
    uncertainty_map = cdflib.xarray.cdf_to_xarray(f'{sys.path[0]}/Anisotropy_DELTA_PLUS-MINUS_newest.cdf')
    uncertainty_map = uncertainty_map.where((uncertainty_map != np.inf) & (uncertainty_map != -1e31))

    angles = uncertainty_map['Variable_4'].values
    anisotropies = uncertainty_map['Variable_5'].values
    deltaPlus = uncertainty_map['UncertaintyMap DELTA_PLUS'].values
    deltaMinus = uncertainty_map['UncertaintyMap DELTA_MINUS'].values

    TH, AN = np.meshgrid(angles, anisotropies, indexing='ij')

    # now lets define a meshgrid to interpolate the values onto. 
    angleMin = 0
    angleMax = 90
    angleBinWidth = angle_bin_width
    angleBinNumber = int((angleMax - angleMin)/angleBinWidth)

    anisotropyMin = 0.03
    anisotropyMax = 15
    anisotropyBinNumber = anisotropy_bins

    anglelist = np.linspace(angleMin, angleMax, angleBinNumber + 1)
    anisotropylist = np.logspace(np.log10(anisotropyMin), np.log10(anisotropyMax), anisotropyBinNumber + 1)

    # Select only the finite points.
    maskPlus  = np.isfinite(deltaPlus)  #(np.any(np.isnan(deltaPlus.data), axis=1))
    maskMinus = np.isfinite(deltaMinus) #(np.any(np.isnan(deltaMinus.data), axis=1))

    # We want the maximum uncertainty in the bin to be the value.
    d_plus, x_edges, y_edges, _ = binned_statistic_2d(
        TH[maskPlus], AN[maskPlus], deltaPlus[maskPlus], 
        statistic='max', 
        bins=[anglelist, anisotropylist]
    )

    d_minus, x_edges, y_edges, _ = binned_statistic_2d(
        TH[maskMinus], AN[maskMinus], deltaMinus[maskMinus], 
        statistic='max', 
        bins=[anglelist, anisotropylist]
    )

    # Calculate Bin Centers for the Fit
    theta_centers = (x_edges[:-1] + x_edges[1:]) / 2
    ani_centers_log = (np.log10(y_edges[:-1]) + np.log10(y_edges[1:])) / 2
    ani_centers = 10**ani_centers_log

    # Return the 
    return(theta_centers, ani_centers, d_plus, d_minus)


def analytical_uncertainty_model(X, sigma_base, T_limit, exponent):
    """
    Parametric model for SPC Uncertainty.
    """
    theta_deg, A_true = X
    theta_rad = np.radians(theta_deg)
    
    S = np.sin(theta_rad)
    C = np.cos(theta_rad)
    
    # 1. Sensitivity (Geometric Term)
    denom_z = (C**2) + A_true * (S**2)
    dm_dA = -1.0 * (S * C) / (denom_z**2)
    sensitivity = np.abs(dm_dA)
    
    # 2. Resolution Penalty (Low Anisotropy)
    resolution_penalty = 1.0 / np.sqrt(denom_z)
    
    # 3. Aperture Clipping Penalty (High Anisotropy)
    T_x = (S**2) + A_true * (C**2)
    clipping_penalty = 1.0 + (T_x / T_limit)**exponent
    
    # 4. Total Noise Sigma
    sigma_m = sigma_base * resolution_penalty * clipping_penalty
    
    # 5. Predicted Uncertainty (Sigma_A)
    sigma_A = sigma_m / (sensitivity + 1e-9)
    
    # return (sigma_A / A_true) * 100 
    return sigma_A

def delta_plus_minus_value(X):
    plus_params  = [0.01371, 1.12289, 1.64729]
    minus_params = [0.00900, 1.00000, 1.06710]

    dplus  = analytical_uncertainty_model(X, *plus_params)
    dminus = analytical_uncertainty_model(X, *minus_params)

    return(dplus, dminus)

def gen_uncertainty_map3(angle_bin_width, anisotropy_bins):
    # now lets define a meshgrid to interpolate the values onto. 
    angleMin = 0
    angleMax = 90
    angleBinWidth = angle_bin_width
    angleBinNumber = int((angleMax - angleMin)/angleBinWidth)

    anisotropyMin = 0.03
    anisotropyMax = 15
    anisotropyBinNumber = anisotropy_bins

    anglelist = np.linspace(angleMin, angleMax, angleBinNumber + 1)
    anisotropylist = np.logspace(np.log10(anisotropyMin), np.log10(anisotropyMax), anisotropyBinNumber + 1)

    TH, AN = np.meshgrid(anglelist, anisotropylist, indexing='ij')

    X = (TH, AN)

    d_plus, d_minus = delta_plus_minus_value(X)

    return(anglelist, anisotropylist, d_plus, d_minus)





