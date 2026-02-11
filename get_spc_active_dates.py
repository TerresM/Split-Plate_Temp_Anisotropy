import pyspedas
import pytplot
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import xarray as xr

import pickle

file_path = 'e23_elected_dates.pkl'




# --- 1. Perihelion Registry (Encounters 4 - 23) ---
# Approximate center dates for each encounter.
# SPC usually hits thermal cutoff 1-5 days before these dates depending on distance.
perihelions = {
    # 4:  '2020-01-29',
    # 5:  '2020-06-07',
    # 6:  '2020-09-27',
    # 7:  '2021-01-17',
    # 8:  '2021-04-29',
    # 9:  '2021-08-09',
    # 10: '2021-11-21',
    # 11: '2022-02-25',
    # 12: '2022-06-01',
    # 13: '2022-09-06',
    # 14: '2022-12-11',
    # 15: '2023-03-17',
    # 16: '2023-06-22',
    # 17: '2023-09-27',
    # 18: '2023-12-29',
    # 19: '2024-03-30',
    # 20: '2024-06-30',
    # 21: '2024-09-30',
    # 22: '2024-12-24', 
    23: '2025-03-22' 
}

def find_thermal_cutoff_and_get_dates(encounter_num, perihelion_date):
    """
    Loads L3 density to find the exact timestamp SPC hits thermal cutoff.
    Returns a list of 5 date strings IMMEDIATELY PRECEDING that cutoff.
    """
    print(f"--- Analyzing Encounter {encounter_num} ({perihelion_date}) ---")
    
    # Search Window:
    # Look from 10 days before perihelion up to perihelion.
    # We exclude post-perihelion to ensure we don't accidentally pick up the restart.
    t_center = datetime.strptime(perihelion_date, "%Y-%m-%d")
    t_start = (t_center - timedelta(days=10)).strftime("%Y-%m-%d")
    t_end   = (t_center).strftime("%Y-%m-%d") 
    
    try:
        # Load L3i Moments (Fast/Lightweight)
        # We only need density ('np_moment') to check for validity
        # 'time_clip=True' is crucial to ensure we don't get data outside our specific inbound window
        pyspedas.psp.spc(
            trange=[t_start, t_end], 
            level='l3i', 
            varnames=['np_moment'], 
            time_clip=True, 
            last_version=True
        )
        
        # Get data from tplot
        times, data = pytplot.get_data('psp_spc_np_moment')
        
        if times is None or len(times) == 0:
            print(f"  [!] Warning: No data found for Enc {encounter_num}")
            return []

        # Convert to pandas for easy handling
        df = pd.Series(data, index=times)
        
        # Filter for Valid Data (L3 fill values are often -1e31 or NaN)
        df_valid = df[df > 0] 
        
        if df_valid.empty:
            print("  [!] No valid density data found in window.")
            return []
            
        # The Thermal Cutoff is the LAST timestamp in this inbound window
        last_valid_time = df_valid.index[-1]
        cutoff_dt = datetime.fromtimestamp(last_valid_time)
        
        print(f"  > Thermal Cutoff detected: {cutoff_dt}")
        
        # Generate the list of 5 days BEFORE this cutoff date
        # Logic: If cutoff is Sept 27th, we want Sept 22, 23, 24, 25, 26.
        target_dates = []
        for i in range(1, 6): # 1 to 5 days back
            d = cutoff_dt - timedelta(days=i)
            target_dates.append(d.strftime("%Y-%m-%d"))
            
        target_dates.sort()
        # print(f"  > Selected Dates: {target_dates}")
        
        return target_dates

    except Exception as e:
        print(f"Error analyzing Enc {encounter_num}: {e}")
        return []

if __name__ == "__main__":
    # Ensure pyspedas saves to the correct local directory
    pyspedas.psp.config.local_data_dir = 'psp_data'
    
    all_process_dates = []
    
    # Loop through the encounters
    for enc, date in perihelions.items():
        # Skip future dates if they exceed current data availability
        if datetime.strptime(date, "%Y-%m-%d") > datetime.now():
            print(f"--- Skipping Encounter {enc} (Future) ---")
            continue
        print(enc, date)
        dates = find_thermal_cutoff_and_get_dates(enc, date)
        all_process_dates.extend(dates)
        
        # Clear memory to prevent pytplot bloat
        pytplot.del_data('*')

    # Sort final list
    final_unique_dates = sorted(list(set(all_process_dates)))
    
    # print("\n" + "="*40)
    # print("COPY THIS LIST INTO 'dates_to_process':")
    # print("="*40)
    # print(final_unique_dates)

    # Save the list (open file in 'wb' binary write mode)
    with open(file_path, 'wb') as f:
        pickle.dump(final_unique_dates, f)

    # # Load the list (open file in 'rb' binary read mode)
    with open(file_path, 'rb') as f:
        loaded_list = pickle.load(f)

    print(loaded_list)

    
