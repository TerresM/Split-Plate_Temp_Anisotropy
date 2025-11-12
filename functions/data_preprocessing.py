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

def xarray_clean_variable(data_array):
    # Get the fill value, valid min, and valid max
    fillval   = data_array.attrs['FILLVAL']
    valid_min = data_array.attrs['VALIDMIN']
    valid_max = data_array.attrs['VALIDMAX']

    # Set all outliers to be NaN.
    data_array.data[data_array.data == fillval] = np.nan
    data_array.data[data_array.data > valid_max] = np.nan
    data_array.data[data_array.data < valid_min] = np.nan

    return(data_array)

def clean_full_xarray(data_xarray):
    fill_mask = np.array(['FILLVAL' in data_xarray[key].attrs for key in data_xarray.data_vars])
    min_mask  = np.array(['VALIDMIN' in data_xarray[key].attrs for key in data_xarray.data_vars])
    max_mask  = np.array(['VALIDMAX' in data_xarray[key].attrs for key in data_xarray.data_vars])
    type_mask = np.array([data_xarray[key].data.dtype == np.float32 for key in data_xarray.data_vars])
    mask = fill_mask & min_mask & max_mask & type_mask

    var_list = np.array(list(data_xarray.data_vars.keys()))[mask]

    for var in var_list:
        data_xarray[var] = xarray_clean_variable(data_xarray[var])

    return data_xarray

def median_filter_current(a_current, b_current, c_current, d_current):
    current_stack = np.stack([a_current, b_current, c_current, d_current])

    # set all fillvals to NaNs
    current_stack[current_stack < 0] = np.nan

    median_current = np.median(current_stack, axis=0)

    for i in range(4):
        current_stack[i][current_stack[i] > 3*median_current] = median_current[current_stack[i] > 3*median_current]

    return(current_stack)