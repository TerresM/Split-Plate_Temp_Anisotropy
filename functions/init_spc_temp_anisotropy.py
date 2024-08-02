import numpy as np
import xarray as xr
import pyspedas 
import pytplot
import os

# user defined functions
from functions.data_preprocessing import clean_variable as clean

def get_relevant_data(trange, credentials, DATA_DIR=None):
    if DATA_DIR:
        os.environ['SPEDAS_DATA_DIR'] = (DATA_DIR)
    
    psp_spc = pyspedas.psp.spc(trange,
                            datatype='l2i',
                            level='L2',
                            notplot=False,
                            get_support_data=True,
                            last_version=True,
                            #no_update=True,
                            username=credentials['psp']['sweap']['username'],
                            password=credentials['psp']['sweap']['password'])
    
    psp_spc_l3 = pyspedas.psp.spi(trange,
                            datatype='spi_sf00',
                            notplot=False,
                            get_support_data=True,
                            last_version=True,
                            #no_update=True,
                            username=credentials['psp']['sweap']['username'],
                            password=credentials['psp']['sweap']['password'])
    
    psp_bfield = pyspedas.psp.fields(trange,
                                     datatype='mag_SC_4_Sa_per_Cyc',
                                     level='l2',
                                     notplot=False,
                                     get_support_data=True,
                                     last_version=True,
                                     #no_update=True,
                                     username=credentials['psp']['fields']['username'],
                                     password=credentials['psp']['fields']['password'])

    return()



