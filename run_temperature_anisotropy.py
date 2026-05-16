import pickle
import pyspedas
import matplotlib.pyplot as plt

from tqdm import tqdm

from support_functions.gen_spc_uncert_map_2d import gen_uncertainty_map_with_bias, gen_uncertainty_map_final # gen_uncertainty_map_newversion    # gen_uncertainty_map2

from functions import spc_processing_pipeline as spc_proc

plt.ion()

if __name__ == "__main__":
    pyspedas.psp.config.local_data_dir = 'psp_data'
    pyspedas.psp.config.CONFIG['sweap_remote_data_dir'] = 'https://w3sweap.cfa.harvard.edu/data/sci/'#'http://131.142.18.203/data/sci/'

    # Configuration
    config = {
        'fit_seconds':4,
        'smooth_seconds': None,
        'spc_mode': 'peak_track',
        'no_weights': None,
        'win_max': False,
        'min_flux': 0.5,
        'avg_bfield': False,
        'data_vars': {
            'spc' : ['l2i', 'l3i'],
            'fields' : 'mag_SC_4_Sa_per_Cyc',
            'spi_sf00' : ['L3']
        },
        'credentials': None, 
        'uncert_map': gen_uncertainty_map_with_bias(), 
        'gen_flag': False,
        'OFFLINE': False, 
    }
    
    file_path = 'selected_dates_updated.pkl'

    # Load the list (open file in 'rb' binary read mode)
    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    dates_to_process = ['2023-09-26']#['2023-12-27']
    tstart = f'{dates_to_process[0]}T05:50:00'
    tend   = f'{dates_to_process[0]}T10:05:00'

    print(f"Queued {len(dates_to_process)} days for processing.")
    
    for date_str in tqdm(dates_to_process):
        spc_proc.process_anisotorpy_spc_event(date_str, config, subrange=[tstart, tend])

    print('All events completed!')
