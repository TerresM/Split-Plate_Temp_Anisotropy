import pyspedas
import cdflib
import cdflib.xarray
import xarray as xr

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

if __name__ == '__main__':
    # Set the time range that we are interested in
    tstart = '2023-09-26T00:00:00' #datetime.datetime(2023, 12, 25, 0, 0, 0)
    tend   = '2023-09-26T12:00:00' #datetime.datetime(2023, 12, 25, 23, 59, 59)
    trange = [tstart, tend]

    # Needed data
    data_vars = {'spc' : ['l2i', 'l3i'],
                 'fields' : 'mag_SC_4_Sa_per_Cyc',
                 'spi_sf00' : ['L3']}

    results = load_data(mission='PSP', datasets=data_vars, trange=trange)