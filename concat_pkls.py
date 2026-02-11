import pickle
import numpy as np


x = np.arange(11, 23)

master_list = {}
for i in x:
    file_path = f'e{i}_elected_dates.pkl'

    with open(file_path, 'rb') as f:
        loaded_list = pickle.load(f)

    master_list[str(i)] = loaded_list