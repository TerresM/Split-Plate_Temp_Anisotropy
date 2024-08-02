import pyspedas
import pytplot
import xarray as xr

# user defined functions
from functions.data_preprocessing import clean_variable as clean

def generate_xarray(trange=None, credentials=None, INCLUDE_DELTA=True, MODE_SELECT=0):
    """
    This subroutine relies on pytplot and the defined pytplot.tnames()

    TODO: 
    """
    if pytplot.tnames() == []:
        if trange == None:
            print("Data set is not downloaded. provide a trange")
        if trange:
            print(f"Data set is not downloaded. \n Downloading data for {trange}")
            if credentials:
                psp_spc = pyspedas.psp.spc(trange,
                                           datatype='l2i',
                                           level='L2',
                                           notplot=False,
                                           get_support_data=True,
                                           last_version=True,
                                           #no_update=True,
                                           username=credentials['psp']['sweap']['username'],
                                           password=credentials['psp']['sweap']['password'])
                MODE = pytplot.data_quants['spp_spc_MODE_FLAG']
                spc_tvars = ['spp_spc_a_current',
                             'spp_spc_b_current',
                             'spp_spc_c_current',
                             'spp_spc_d_current',
                             'spp_spc_diff_charge_flux_density']
                print('Here')
            else:
                psp_spc = pyspedas.psp.spc(trange,
                                           datatype='l2i',
                                           level='l2',
                                           notplot=False,
                                           get_support_data=True,
                                           last_version=True,
                                           #no_update=True
                                           )
                MODE = pytplot.data_quants['psp_spc_MODE_FLAG']
                spc_tvars = ['psp_spc_a_current',
                             'psp_spc_b_current',
                             'psp_spc_c_current',
                             'psp_spc_d_current',
                             'psp_spc_diff_charge_flux_density']       
                print("Try Here")         

    #QFs = pytplot.data_quants['spp_spc_DQF']
    #MODE = pytplot.data_quants['spp_spc_MODE_FLAG']

    spc_tvar_names = ['a_current', 'b_current', 'c_current', 'd_current', 'diff_charge_flux']
    print(spc_tvars)

    if INCLUDE_DELTA:
        if credentials:
            spc_tvars_DELTA = ['spp_spc_a_current_DELTA_VAR',
                               'spp_spc_b_current_DELTA_VAR',
                               'spp_spc_c_current_DELTA_VAR',
                               'spp_spc_d_current_DELTA_VAR',
                               'spp_spc_diff_charge_flux_density_DELTA_VAR']
        else:
            spc_tvars_DELTA = ['psp_spc_a_current_DELTA_VAR',
                               'psp_spc_b_current_DELTA_VAR',
                               'psp_spc_c_current_DELTA_VAR',
                               'psp_spc_d_current_DELTA_VAR',
                               'psp_spc_diff_charge_flux_density_DELTA_VAR']
        
        spc_tvar_names_DELTA = ['a_current_DELTA', 'b_current_DELTA', 'c_current_DELTA', 'd_current_DELTA', 'diff_charge_flux_DELTA']

        spc_tvars = spc_tvars + spc_tvars_DELTA
        spc_tvar_names = spc_tvar_names + spc_tvar_names_DELTA

    tvar_list = []
    for var in spc_tvars:
        tvar_list.append(clean(pytplot.data_quants[var]))

    spc_var = xr.merge(tvar_list).rename(dict(zip(spc_tvars, spc_tvar_names)))

    # Drop redundant v and spec_bins
    spc_var = spc_var.drop_vars(['v', 'spec_bins'])

    # Assign mv_lo to the coordinates
    if credentials:
        mv_lo = clean(pytplot.data_quants['spp_spc_mv_lo']).drop_vars(['v', 'spec_bins'])
        mv_hi = clean(pytplot.data_quants['spp_spc_mv_hi']).drop_vars(['v', 'spec_bins']).rename('mv_hi')
    else:
        mv_lo = clean(pytplot.data_quants['psp_spc_mv_lo']).drop_vars(['v', 'spec_bins'])
        mv_hi = clean(pytplot.data_quants['psp_spc_mv_hi']).drop_vars(['v', 'spec_bins']).rename('mv_hi')

    spc_var = xr.merge([spc_var, mv_hi])
    spc_var = spc_var.assign_coords({'mv_lo' : mv_lo}) #, 'mv_hi' : mv_hi})
    
    if INCLUDE_DELTA:
        if credentials:
            mv_lo_DELTA = clean(pytplot.data_quants['spp_spc_mv_lo_DELTA_VAR']).drop_vars(['v', 'spec_bins']).rename('mv_lo_DELTA')
            mv_hi_DELTA = clean(pytplot.data_quants['spp_spc_mv_hi_DELTA_VAR']).drop_vars(['v', 'spec_bins']).rename('mv_hi_DELTA')
        else:
            mv_lo_DELTA = clean(pytplot.data_quants['psp_spc_mv_lo_DELTA_VAR']).drop_vars(['v', 'spec_bins']).rename('mv_lo_DELTA')
            mv_hi_DELTA = clean(pytplot.data_quants['psp_spc_mv_hi_DELTA_VAR']).drop_vars(['v', 'spec_bins']).rename('mv_hi_DELTA')
        spc_var = xr.merge([spc_var, mv_lo_DELTA, mv_hi_DELTA])

    # Drop the energy channels above 32!
    spc_var = spc_var.isel(v_dim=slice(0,32))

    if MODE_SELECT == 0:
        print('Selecting only peak tracking data!')

        # Drop all periods not in peak tracking mode!
        spc_var = spc_var.where(MODE == 0, drop=True)

    return(spc_var) 