import numpy as np
import xarray as xr

def clean_variable(var):
    """
    Cleans an xarray.DataArray or a single variable of an xarray.Dataset.

    NOTE: This works well for PSP data and is built as an extension for the 
    pyspedas data routine.

    Parameters:
    -----------
    var : xarray
            Xarray dataset containing meta data
    
    Returns:
    -------
    clean : xarray
            The cleaned version of the xarray. Removed bad data!
    """

    # Get the fill value, valid min, and valid max from meta data
    fillval = var.CDF['VATT']['FILLVAL']
    valid_min = var.CDF['VATT']['VALIDMIN']
    valid_max = var.CDF['VATT']['VALIDMAX']

    # set data values above the min, max, and fill values to be nan's
    var.data[var.data == fillval]  = np.nan
    var.data[var.data > valid_max] = np.nan
    var.data[var.data < valid_min] = np.nan

    clean = var

    return(clean)