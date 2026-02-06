import numpy as np
import matplotlib.pyplot as plt
import scipy
import pyspedas
import cdflib
import xarray


# March 30, 2024 00:00 UT – April 5, 2024 00:00 UT
trange = ['2024-03-30T00:00:00', '2024-04-05T00:00:00']

psp_bfile = pyspedas.psp.fields(trange,
                                datatype=f'mag_SC_1min',
                                level='l2',
                                notplot=True,
                                get_support_data=True,
                                last_version=True,
                                downloadonly=True)

span_file = pyspedas.psp.spi(trange, datatype='spi_sf00_l3_mom', level='l3', notplot=True, time_clip=True, downloadonly=True, last_version=True)
