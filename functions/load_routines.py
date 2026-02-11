import os
import glob
import pyspedas
import cdflib
import cdflib.xarray
import xarray as xr

import numpy as np
try:
    from . import load_config
except:
    import load_config

from datetime import datetime

def _load_psp_fields_file(trange, dtype, creds):
    if creds:
        psp_bfile = pyspedas.psp.fields(trange,
                                        datatype=dtype,
                                        downloadonly=True,
                                        notplot=True,
                                        get_support_data=True,
                                        last_version=True,
                                        username=creds['psp']['fields']['username'],
                                        password=creds['psp']['fields']['password'])
    else:
        psp_bfile = pyspedas.psp.fields(trange,
                                        datatype=dtype,
                                        downloadonly=True,
                                        notplot=True,
                                        get_support_data=True,
                                        last_version=True)
    return psp_bfile

def _load_psp_spc_file(trange, dtype, creds):
    if creds:
        psp_spc_file = pyspedas.psp.spc(trange,
                                        datatype=dtype,
                                        notplot=True,
                                        downloadonly=True,
                                        get_support_data=True,
                                        last_version=True,
                                        username=creds['psp']['sweap']['username'],
                                        password=creds['psp']['sweap']['password'])
    else:
        psp_spc_file = pyspedas.psp.spc(trange,
                                        datatype=dtype,
                                        notplot=True,
                                        downloadonly=True,
                                        get_support_data=True,
                                        last_version=True)    
    return psp_spc_file

def _load_psp_spi_file(trange, dtype, level=None, creds=None):
    if creds:
        psp_spi_file = pyspedas.psp.spi(trange,
                                        datatype=dtype,
                                        level=level,
                                        notplot=True,
                                        downloadonly=True,
                                        get_support_data=True,
                                        last_version=True,
                                        username=creds['psp']['sweap']['username'],
                                        password=creds['psp']['sweap']['password'])
    else:
        psp_spi_file = pyspedas.psp.spi(trange,
                                        datatype=dtype,
                                        notplot=True,
                                        downloadonly=True,
                                        get_support_data=True,
                                        last_version=True)    
    return psp_spi_file

def load_data(mission='psp', datasets=None, trange=None, creds = None, help=False):
    '''
    Flexible data loader for pyspedas that can handle dictionary-based
    dataset collection.

    Parameters:
    -----------


    Returns:
    --------

    '''
    if help:
        dict = {'fields' : ['mag_RTN',
                            'mag_RTN_1min',
                            'mag_rtn_4_per_cycle',
                            'mag_RTN_4_Sa_per_Cyc', 
                            'mag_SC',
                            'mag_SC_1min',
                            'mag_SC_4_Sa_per_Cyc', 
                            'mag_VSO',
                            'rfs_burst',
                            'rfs_hfr', 
                            'rfs_lfr',
                            'f2_100bps',
                            'dfb_dc_spec',
                            'dfb_ac_spec',
                            'dfb_dc_xspec',
                            'dfb_ac_xspec',
                            'merged_scam_wf',
                            'sqtn_rfs_V1V2']
                }
        print('This is under construction.')
        return()
    
    if datasets is None or trange is None:
        raise ValueError('Must provide datasets and trange unless help=True.')
    
    mission = mission.lower()   # ensure everything is lower case.
    results = {}    # The result dictionary.

    if mission == "psp":
        for inst, dset in datasets.items():
            if inst == 'spc':
                if isinstance(dset, str):   # if only one value make it a list
                    dset = [dset]
                results['spc'] = {}
                for lvl in dset:
                    if creds:
                        results['spc'][lvl]= _load_psp_spc_file(trange, dtype=f'{lvl}', creds=creds)
                    else:
                        results['spc'][lvl]= _load_psp_spc_file(trange, dtype=f'{lvl}', creds=None)
            elif inst == 'fields':
                if isinstance(dset, str):   # if only one value make it a list
                    dset = [dset]
                results['fields'] = {}
                for lvl in dset:
                    if creds:
                        results['fields'][lvl]= _load_psp_fields_file(trange, dtype=f'{lvl}', creds=creds)
                    else:
                        results['fields'][lvl]= _load_psp_fields_file(trange, dtype=f'{lvl}', creds=None)
            elif inst == 'spi' and creds == None:
                if isinstance(dset, str):   # if only one value make it a list
                    dset = [dset]
                results['spi'] = {}
                for lvl in dset:
                    results['spi'][lvl]= _load_psp_spi_file(trange, dtype=f'{lvl}', creds=None)
            elif inst == 'spi_sf00':
                    if creds:
                        if isinstance(dset, str):   # if only one value make it a list
                            dset = [dset]
                        results['spi_sf00'] = {}
                        for lvl in dset:
                            print('Loading in the spi data with creds!')
                            results['spi_sf00'][lvl]= _load_psp_spi_file(trange, dtype='spi_sf00', level=lvl, creds=creds)
                    else:
                        ValueError('Must provide credentials if loading in instrument team data!')
                        

    return(results)

def file_to_xarray(files):
    if isinstance(files, list) and len(files) > 1:
        file_list = [cdflib.xarray.cdf_to_xarray(files[i]) for i in range(len(files))]
        time_candidates = [c for c in file_list[0].coords if 'epoch' in c.lower()]

        time_coord = time_candidates[0]     # It should be the first coord

        if len(time_coord) > 1:
            file_list = [file_list.rename({f'{time_coord}':'epoch'}).drop_dim(time_coord[1]) for i in range(len(file_list))]
        else:
            file_list = [file_list.rename({f'{time_coord}':'epoch'}) for i in range(len(file_list))]

        file_xr   = xr.concat(file_list, dim='epoch')
    else:
        if isinstance(files, str):
            files = [files]
        
        file_xr = cdflib.xarray.cdf_to_xarray(files[0], fillval_to_nan=True)

        time_candidates = [c for c in file_xr.coords if 'epoch' in c.lower()]

        time_coord = time_candidates[0]     # It should be the first coord

        file_xr = file_xr.rename({f'{time_coord}':'epoch'})

    return file_xr

def load_local_data_file(trange, datasets, creds):
    # This function is specific to the temperature anisotropy load routine. 
    # The supplied trange is used to extract the local string. 

    # This function will load in the local SPC l2, l3, and supporting fields file for
    # the defined time range. 

    # Step 1 define the total number of days
    # Trange is defined as a list of datetime strings (i.e. '2025-01-01T01:00:00')
    # Convert to datetime64, take difference, take floor, always set to be 1 day.
    num_days = np.floor((np.datetime64(trange[1]) - np.datetime64(trange[0]))/np.timedelta64(1,'D')) + 1   # The +1 ensures atleast 1 day

    # Extract the key time information that is needed for the folder searching. 
    dt_object = datetime.strptime(trange[0], '%Y-%m-%dT%H:%M:%S')

    year, month, day = dt_object.year, dt_object.month, dt_object.day
    hour, mins, secs = dt_object.hour, dt_object.min, dt_object.second

    source_directory = load_config.source_data_directory
    print(f'Loading from local data directory: {source_directory}')
    # Now that we have the source data directory and file stems we can add the files to the xarray lists.
    # Look at the required datasets. This is a dictonary.
    keys = set(datasets.keys())
    allowed_keys = {'spc', 'fields', 'spi_sf00'}

    invalid_keys = keys - allowed_keys

    if invalid_keys:
        raise ValueError(f"Invalid data_vars keys found: {invalid_keys}. "
                     f"Allowed keys are: {allowed_keys}")
    
    results = {}
    for inst, dset in datasets.items():
        yyyymmdd = f'{year}{month:02}{day:02}'
        if inst == 'spc':
            if isinstance(dset, str):   # if only one value make it a list
                dset = [dset]
            results['spc'] = {}
            for lvl in dset:
                if creds:
                    file_str_start = f'spp_swp_spc_{lvl}_{yyyymmdd}_v**.cdf'
                    if lvl == 'l2i': lvl_str = 'L2'
                    if lvl == 'l3i': lvl_str = 'L3'
                    # path = source_directory+f'/{lvl_str}/{year}/{month:02}/'
                    search_path = os.path.join(source_directory, 'sweap', inst, lvl_str, str(year), f'{month:02}', file_str_start)
                    found_file = glob.glob(search_path)

                    print('here', found_file)
                    found_file.sort()
                    found_file = found_file[-1]
                else:
                    file_str_start = f'psp_swp_spc_{lvl}_{yyyymmdd}_v**.cdf'
                    if lvl == 'l2i': lvl_str = 'l2'
                    if lvl == 'l3i': lvl_str = 'l3'
                    # path = source_directory+f'/{lvl_str}/{lvl}/{year}/'
                    search_path = os.path.join(source_directory, 'sweap', inst, lvl_str, lvl, str(year), file_str_start)
                    found_file = glob.glob(search_path)
                    found_file.sort()
                    found_file = found_file[-1]

                results['spc'][lvl] = found_file
        elif inst == 'fields':
            if isinstance(dset, str):   # if only one value make it a list
                dset = [dset]
            results['fields'] = {}
            for lvl in dset:
                if creds:
                    file_str_start = f'psp_fld_l2_mag_SC_4_Sa_per_Cyc_{yyyymmdd}_v**.cdf'
                    search_path = os.path.join(source_directory, 'fields', 'l2', lvl, str(year), f'{month:02}', file_str_start)
                    found_file = glob.glob(search_path)
                    found_file.sort()
                    found_file = found_file[-1]
                else:
                    file_str_start = f'psp_fld_l2_mag_sc_4_sa_per_cyc_{yyyymmdd}_v**.cdf'
                    search_path = os.path.join(source_directory, 'fields', 'l2', 'mag_sc_4_per_cycle', str(year), file_str_start)
                    found_file = glob.glob(search_path)
                    found_file.sort()
                    found_file = found_file[-1]
            
            results['fields'][lvl] = found_file
        elif inst == 'spi_sf00':
            if creds:
                if isinstance(dset, str):   # if only one value make it a list
                    dset = [dset]
                results['spi_sf00'] = {}
                for lvl in dset:
                    print('Loading in the spi data with creds!')
                    file_str_start = f'psp_swp_spi_sf00_L3_mom_{yyyymmdd}_v**.cdf'
                    search_path = os.path.join(source_directory, 'sweap', 'spi', 'L3', inst, str(year), f'{month:02}', file_str_start)
                    found_file = glob.glob(search_path)
                    found_file.sort()
                    found_file = found_file[-1]
            else:
                ValueError('Must provide credentials if loading in instrument team data!')

            results['spi_sf00'][lvl] = found_file

    return results


if __name__ == '__main__':
    # Set the time range that we are interested in
    tstart = '2023-12-27T00:00:00' #datetime.datetime(2023, 12, 25, 0, 0, 0)
    tend   = '2023-12-27T12:00:00' #datetime.datetime(2023, 12, 25, 23, 59, 59)
    trange = [tstart, tend]

    # Needed data
    data_vars = {'spc' : ['l2i', 'l3i'],
                 'fields' : 'mag_SC_4_Sa_per_Cyc',
                 'spi_sf00' : ['L3']}

    # results = load_local_data_file(trange, data_vars, creds=True)
    results = load_data(mission='PSP', datasets=data_vars, trange=trange)