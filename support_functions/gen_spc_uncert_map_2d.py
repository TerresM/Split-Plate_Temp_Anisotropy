import os
import sys
import scipy
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
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

