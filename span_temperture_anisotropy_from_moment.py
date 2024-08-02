import numpy as np
import pyspedas
import datetime
import cdflib
import pandas as pd
import xarray as xr
import json


import matplotlib.pyplot as plt

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def convert_to_tensor(input_array):
    N_time = input_array.shape[0]
    tensors = np.zeros((N_time, 3, 3))

    for i in range(N_time):
        T_input = input_array[i]
        tensors[i] = np.array([[T_input[0], T_input[3], T_input[4]],
                               [T_input[3], T_input[1], T_input[5]],
                               [T_input[4], T_input[5], T_input[2]]])

    return tensors

def define_time_series_rot_vector(bvec):
    Ntime = bvec.shape[0]

    # Get the unit vector
    bunit = bvec / np.linalg.norm(bvec, axis=1)[:,None]

    # Get the 
    temp = np.stack([np.zeros(Ntime), np.ones(Ntime), np.zeros(Ntime)]).T

    # Define a vector normal to the magnetic field unit vector
    P = np.cross(bunit, temp)/np.linalg.norm(np.cross(bunit, temp), axis=1)[:,None]

    # Complete the matrix
    Q = np.cross(bunit, P)/np.linalg.norm(np.cross(bunit, P), axis=1)[:,None]

    # Define the rotation matrix
    R = np.stack([bunit, P, Q], axis=1)

    return(R)

def rotate_the_magnetic_field_vector(t_tensor, rot):
    
    Ntime = len(t_tensor)

    T_rot = []
    for i in range(Ntime):
        A = np.matmul(t_tensor[i], np.linalg.inv(rot[i]))
        T = np.matmul(rot[i], A)
        T_rot.append(T)

    T_rot = np.array(T_rot)

    return(T_rot)




if __name__ == "__main__":
    # Set the time range that we are interested in
    tstart = datetime.datetime(2023, 9, 26)
    tend   = datetime.datetime(2023, 9, 27)
    trange = [tstart, tend]
    trange_bfield = [tstart - datetime.timedelta(days= 1), tend]

    date_str = str(tstart)[0:10] 

    credentials = load_config('/home/michael/Research/pySolarWind/config.json')
    
    psp_bfield = pyspedas.psp.fields(trange_bfield,
                                     datatype='mag_SC_1min',
                                     level='l2',
                                     notplot=False,
                                     get_support_data=True,
                                     last_version=True,
                                     username=credentials['psp']['fields']['username'],
                                     password=credentials['psp']['fields']['password'])
    
    psp_spc_l3  = pyspedas.psp.spc(trange=trange,
                                   datatype='l3i',
                                   level='L3',
                                   notplot=True,
                                   downloadonly=True,
                                   get_support_data=True,
                                   last_version=True,
                                   username=credentials['psp']['sweap']['username'],
                                   password=credentials['psp']['sweap']['password'])

    psp_span_l3 = pyspedas.psp.spi(trange=trange,
                                   datatype='spi_sf00',
                                   level='L3',
                                   notplot=True,
                                   downloadonly=True,
                                   get_support_data=True,
                                   last_version=True,
                                   username=credentials['psp']['sweap']['username'],
                                   password=credentials['psp']['sweap']['password'])
    
    span_l3_xr = cdflib.cdf_to_xarray(psp_span_l3[0], to_datetime=True)

    T_tensor = span_l3_xr.T_TENSOR_INST.data
    T_tens_form = convert_to_tensor(T_tensor)

    B_vec_inst = span_l3_xr.MAGF_INST.data

    # Smoothed B-field
    window = 18

    Bx_smooth = np.convolve(B_vec_inst[:,0], np.ones(window)/window, 'same')
    By_smooth = np.convolve(B_vec_inst[:,1], np.ones(window)/window, 'same')
    Bz_smooth = np.convolve(B_vec_inst[:,2], np.ones(window)/window, 'same')

    smooth_bvec_inst = np.stack([Bx_smooth, By_smooth, Bz_smooth]).T

    R = define_time_series_rot_vector(smooth_bvec_inst)

    T_bframe = rotate_the_magnetic_field_vector(T_tens_form, R)
