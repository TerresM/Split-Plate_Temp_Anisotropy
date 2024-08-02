import numpy as np
import xarray as xr
from tqdm import tqdm

import functions.spc_calc_anisotropy_v2 as calc


def perform_fitting_with_time_windows(groups, l2_data, l3_data, fields_data, 
                                         uncert_map, VERBOSE=True, AS_XARRAY=True,
                                         SUPPORT=False, PEAK_NORM=False, 
                                         TOTAL_CURRENT=False, NO_WEIGHTS = False, 
                                         PLOT=False):
    """
    groups : list 
        contains the time slices that are being used in the analysis. 

    l2_data : xr.dataset 
        contains the main fitted data.

    l3_data : xr.dataset 
        contains the corresponding l3 fits. Helpful support data.

    fields_data : 
    """
    # It is significantly quicker to assign a list to store the data
    fitting_list = []
    #fields_list = []
    support_list = []

    nan_count = 0

    # Loop over all the groups
    for i in tqdm(range(len(groups))):
        # Define the timestamp
        middle_time = groups[i][0] + np.diff(groups[i])/2

        # select the data tha falls inside our window.
        sel_l2_ds = l2_data.sel(time = slice(*groups[i]))
        sel_flds  = fields_data.sel(time = slice(*groups[i]))
        
        # Check if the selected subset has all the data!
        if sel_l2_ds.dims['time'] > 0:
            # TODO: Perhaps add restriction to have at least two spectra to perform fitting
            # If in this loop data exists!
            sel_l3_ds = l3_data.sel(time = slice(*groups[i]))

            sel_l3_vals = sel_l3_ds.mean(skipna=True)
            sel_l3_vals['timestamp'] = middle_time

            # Now we need to check that the b-field data exists!
            if np.any(np.isnan(sel_flds)):
                # if there is a missing magnetic field vector assign NaN values
                l2_fitting_result = calc._spc_nan_anisotropy(middle_time)
                nan_count +=1
            else:
                l2_fitting_result = calc.spc_calculate_anisotropy(middle_time, sel_l2_ds, sel_flds, uncert_map, threshold=0.5,
                                                                   PEAK_NORM=PEAK_NORM, TOTAL_CURRENT=TOTAL_CURRENT, NO_WEIGHTS=NO_WEIGHTS)

            fitting_list.append(l2_fitting_result)
            support_list.append(sel_l3_vals)
        
        # if there is no time selected for this subset!
        else:
            l2_fitting_result = calc._spc_nan_anisotropy(middle_time)
            nan_count +=1

            fitting_list.append(l2_fitting_result)

    
    if VERBOSE:
        print(f"A total of {len(groups)} groups were fit.")
        print(f"Of those groups {nan_count} were filled with NaN values.")

    if AS_XARRAY:
        # It is faster to concatenate all the lists after completing the main loop!
        fitting_result = xr.concat(fitting_list, dim='time')
        
    else:
        fitting_result = fitting_list
    
    
    if SUPPORT:
        return(fitting_result, support_list)
    else:
        return(fitting_result)


# TODO : NEED TO FIX THIS FUNCTION. USE ONLY THE WITH WINDOWS
def perform_fitting_with_groups(xr_groups, l2_data, l3_data, fields_data, uncert_map, NORM=False, VERBOSE=True, AS_XARRAY=True, SUPPORT=False, PLOT=False):
    """
    groups : list 
        contains the time slices that are being used in the analysis. 

    l2_data : xr.dataset 
        contains the main fitted data.

    l3_data : xr.dataset 
        contains the corresponding l3 fits. Helpful support data.

    fields_data : 
    """

    time_label = list(xr_groups.keys())
    groups = list(xr_groups.values())

    # It is significantly quicker to assign a list to store the data
    fitting_list = []
    support_list = []

    nan_count = 0

    # Loop over all the groups
    for i in tqdm(range(len(groups))):
        # Define the timestamp
        middle_time = time_label[i]

        # select the data tha falls inside our window.
        sel_l2_ds = l2_data.isel(time = groups[i])
        sel_flds  = fields_data.isel(time = groups[i])
                
        # TODO: Perhaps add restriction to have at least two spectra to perform fitting
        # If in this loop data exists!
        sel_l3_ds = l3_data.isel(time = groups[i])

        sel_l3_vals = sel_l3_ds.mean(skipna=True)
        sel_l3_vals['timestamp'] = middle_time

        # Now we need to check that the b-field data exists!
        if np.any(np.isnan(sel_flds)):
            # if there is a missing magnetic field vector assign NaN values
            l2_fitting_result = calc._spc_nan_anisotropy(middle_time)
            l3_support = calc.gen_support_file_nan(middle_time)
            nan_count += 1
        else:
            l2_fitting_result = calc.spc_calculate_anisotropy(middle_time, sel_l2_ds, sel_flds, uncert_map, 
                                                                weighted=True, RDF=True, SINGLE_NORMALIZE=NORM,
                                                                DROP=True, PLOT=PLOT)
            
        fitting_list.append(l2_fitting_result)
        support_list.append(sel_l3_vals)
        
    if VERBOSE:
        print(f"A total of {len(groups)} groups were fit.")
        print(f"Of those groups {nan_count} were filled with NaN values.")

    if AS_XARRAY:
        # It is faster to concatenate all the lists after completing the main loop!
        fitting_result = xr.concat(fitting_list, dim='time')
        
    else:
        fitting_result = fitting_list
    
    
    if SUPPORT:
        return(fitting_result, support_list)
    else:
        return(fitting_result)