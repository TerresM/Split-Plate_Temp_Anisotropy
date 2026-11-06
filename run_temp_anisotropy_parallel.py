import pickle
import pyspedas
import multiprocessing
import traceback
import matplotlib

# Force Matplotlib to run in headless mode. 
# This prevents memory leaks from hidden GUI backends in parallel workers.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your custom functions
from support_functions.gen_spc_uncert_map_2d import gen_uncertainty_map_with_bias
from functions import spc_processing_pipeline as spc_proc

if __name__ == "__main__":
    pyspedas.psp.config.local_data_dir = 'psp_data'
    pyspedas.psp.config.CONFIG['sweap_remote_data_dir'] = 'https://w3sweap.cfa.harvard.edu/data/sci/'

    # 1. Configuration (Updated to match your desired settings)
    config = {
        'fit_seconds': 4,
        'smooth_seconds': None,
        'spc_mode': 'peak_track',
        'no_weights': None,
        'weighted_fit': False,             # Brought over from the serial script
        'win_max': False,
        'min_flux': 0.38,                 # Brought over from the serial script
        'flow_correction': False,
        'avg_bfield': False,
        'data_vars': {
            'spc' : ['l2i', 'l3i'],
            'fields' : 'mag_SC_4_Sa_per_Cyc',
        },
        'credentials': None, 
        'uncert_map': gen_uncertainty_map_with_bias(), 
        'gen_flag': True,
        'OFFLINE': True, 
        'likelihood_method': "T",    # Brought over from the serial script
        'plot_diagnostics': False,
    }
    
    file_path = 'selected_dates_updated_v2.pkl'

    # Load the dates list[cite: 5]
    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    loaded_list = [date for date_list in loaded_dict.values() for date in date_list]

    cutoff = datetime(2022, 4, 1)
    dates_to_process = [
        d for d in loaded_list
        if datetime.strptime(d, "%Y-%m-%d") < cutoff
    ]

    # 2. Memory-Safe Worker Allocation
    # Given 32 GB of RAM and 600MB+ arrays, we want to leave headroom for MCMC spikes.
    # 3 workers x ~8 GB peak per worker = 24 GB (leaves 8 GB for OS operations).
    MAX_WORKERS = 4
    print(f"Queued {len(dates_to_process)} days for processing. Using {MAX_WORKERS} CPU cores.")
    
    worker_func = partial(spc_proc.process_anisotorpy_spc_event, config_dict=config, METHOD='Bayesian_MCMC_5D_NCP')

    # 3. Process Execution with Memory Purging
    # 'spawn' is thread-safe for file I/O. max_tasks_per_child=1 prevents memory leaks.
    context = multiprocessing.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=context, max_tasks_per_child=1) as executor:
        
        # Submit all days to the pool, passing a position ID for tqdm formatting[cite: 5]
        futures = {
            executor.submit(worker_func, date_str, pos_id=(i % MAX_WORKERS)): date_str 
            for i, date_str in enumerate(dates_to_process)
        }
        
        # Track completed days globally[cite: 5]
        for future in tqdm(as_completed(futures), total=len(dates_to_process), desc="Days Completed", position=MAX_WORKERS):
            date_str = futures[future]
            try:
                future.result() 
            except Exception as exc:
                print(f"\n!!! CRITICAL FAILURE on {date_str} !!!")
                print(f"Error: {exc}")
                # Print the full traceback so we can see exactly where it broke
                traceback.print_exception(type(exc), exc, exc.__traceback__)

    print('\nAll events completed!')