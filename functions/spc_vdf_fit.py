import numpy as np
import scipy 
import cdflib
import xarray as xr

import matplotlib.pyplot as plt

def Maxwellian(velocity, den, v, v_th):
    a = den/(v_th*np.sqrt(np.pi))*np.exp(-(velocity - v)**2 / v_th**2)
    return(a)

def TwoMaxwellian(velocity, den, v, v_th, den1, v1, v_th1):
    a = den/(v_th*np.sqrt(np.pi))*np.exp(-(velocity - v)**2 / v_th**2)
    b = den1/(v_th1*np.sqrt(np.pi))*np.exp(-(velocity - v1)**2/v_th1**2)
    return(a + b)

def ThreeMaxwellian(velocity, den, v, v_th, den1, v1, v_th1, den2, v2, v_th2):
    a = den/(v_th*np.sqrt(np.pi))*np.exp(-(velocity - v)**2 / v_th**2)
    b = den1/(v_th1*np.sqrt(np.pi))*np.exp(-(velocity - v1)**2/v_th1**2)
    c = den2/(v_th2*np.sqrt(np.pi))*np.exp(-(velocity - v2)**2/v_th2**2)
    return(a + b + c)

def Fit_SPC_Data():
    return()