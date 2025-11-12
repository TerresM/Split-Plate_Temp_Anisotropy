import json
import numpy as np

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def hampel_filter(input_series, window_size, n_sigmas=3):
    """
    Method to apply a Hampel filter to a given time series to remove spurious data points.
    Parameters
    ----------
    input_series: numpy array
        The time series to be filtered
    window_size: int
        The size of the window to use in the filtering
    n_sigmas: float
        The number of standard deviations used to determine if a data point is a spike

    Returns
    -------
    new_series: numpy array
        The filtered time series
    indices: list
        A list of the indices of the removed data points
    """
    
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    indices = []
    
    for i in range((window_size),(n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices

def hampel_filter2(input_array, window_size, n_sigmas=3):
    """
    Apply a Hampel filter to 1D or 2D numpy arrays to remove spurious data points.
    
    Parameters
    ----------
    input_array : numpy array
        1D time series or 2D array (n_series, n_time).
    window_size : int
        The size of the window to use in the filtering.
    n_sigmas : float
        The number of standard deviations used to determine if a data point is a spike.

    Returns
    -------
    new_array : numpy array
        The filtered array, same shape as input_array.
    indices : list of lists
        A list of spike indices for each series (for 1D, just a single list).
    """
    input_array = np.asarray(input_array)
    k = 1.4826  # scale factor for Gaussian distribution
    
    if input_array.ndim == 1:  # Single series
        input_array = input_array[None, :]  # add fake series dimension
    
    n_series, n_time = input_array.shape
    new_array = input_array.copy()
    all_indices = []
    
    for s in range(n_series):
        series = input_array[s]
        indices = []
        for i in range(window_size, n_time - window_size):
            window = series[i - window_size : i + window_size]
            x0 = np.nanmedian(window)
            S0 = k * np.nanmedian(np.abs(window - x0))
            
            if np.abs(series[i] - x0) > n_sigmas * S0:
                new_array[s, i] = x0
                indices.append(i)
        all_indices.append(indices)
    
    if all_indices and len(all_indices) == 1:  # collapse for 1D input
        all_indices = all_indices[0]
        new_array = new_array[0]
    
    return new_array, all_indices

def fft_hampel_filter(input_series, window_size, n_sigmas=3):
    '''
    Works best with odd window sizes!
    '''
    # Take the FFT of the initial timeseries
    fft_series = np.fft.fft(input_series)

    # apply the hampel filter
    fft_filtered = hampel_filter(fft_series, window_size, n_sigmas=n_sigmas)

    # Get the inverse fft
    filtered_series = np.real(np.fft.ifft(fft_filtered[0]))

    return(filtered_series)