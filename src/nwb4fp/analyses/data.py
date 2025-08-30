from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import nwb4fp.analyses.maps as mapp
import ast
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import os

# 分组函数
def group_channels_by_group(channel_list):
    file_path = r"Q:\sachuriga\Sachuriga_Python/quattrocolo-nwb4fp/ASSY-236-F.prb"
    # Read the file and parse the dictionary
    local_vars = {'np': np}
    with open(file_path, 'r') as file:
        exec(file.read(), local_vars)  # Execute the file content with NumPy in scope

        
    channel_groups = local_vars.get('channel_groups')
    if channel_groups is None:
        raise ValueError(f"'channel_groups' not found in {file_path}")

    # Assuming channel_groups is loaded from Step 1
    data = []
    for group_id, group_data in channel_groups.items():
        channels = group_data['channels']
        geometry = group_data['geometry']
        for channel in channels:
            x, y = geometry[channel]
            data.append({
                'group_id': group_id,
                'channel_id': channel,
                'x': x,
                'y': y
            })
    probe_df = pd.DataFrame(data)
    df= probe_df

    grouped = {}
    for channel in channel_list:
        group = df[df['channel_id'] == channel]['group_id'].values[0]
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(channel)
    return grouped

# 找到每个组的中间 channel_id
def find_middle_channel_per_group(grouped_channels):
    file_path = r"Q:\sachuriga\Sachuriga_Python/quattrocolo-nwb4fp/ASSY-236-F.prb"
    # Read the file and parse the dictionary
    local_vars = {'np': np}
    with open(file_path, 'r') as file:
        exec(file.read(), local_vars)  # Execute the file content with NumPy in scope

        
    channel_groups = local_vars.get('channel_groups')
    if channel_groups is None:
        raise ValueError(f"'channel_groups' not found in {file_path}")

    # Assuming channel_groups is loaded from Step 1
    data = []
    for group_id, group_data in channel_groups.items():
        channels = group_data['channels']
        geometry = group_data['geometry']
        for channel in channels:
            x, y = geometry[channel]
            data.append({
                'group_id': group_id,
                'channel_id': channel,
                'x': x,
                'y': y
            })
    probe_df = pd.DataFrame(data)
    df= probe_df  

    middle_channels = {}
    for group, channels in grouped_channels.items():
        group_df = df[df['channel_id'].isin(channels)]
        sorted_df = group_df.sort_values('y', ascending=False).reset_index(drop=True)
        middle_idx = len(sorted_df) // 2
        middle_channel = sorted_df.iloc[middle_idx]['channel_id']
        middle_channels[group] = middle_channel
    return middle_channels

def get_nearest_8_by_position(numbers, target, bad_channels):
    # Filter out values in bad_channels
    filtered_numbers = [num for num in numbers if num not in bad_channels]
    
    # Handle empty cases
    if not filtered_numbers:
        return []
    
    # If target not found, return up to 8 values, padded if needed
    if target not in filtered_numbers:
        result = filtered_numbers[:8]
        if len(result) < 8 and result:
            result.extend([filtered_numbers[-1]] * (8 - len(result)))
        return result
    
    # Initialize result with target
    result = [target]
    target_idx = filtered_numbers.index(target)
    left = target_idx - 1
    right = target_idx + 1
    direction = 'left'  # Start with left
    
    # Sweep left and right alternately
    while len(result) < 8 and (left >= 0 or right < len(filtered_numbers)):
        if direction == 'left' and left >= 0:
            result.append(filtered_numbers[left])
            left -= 1
            direction = 'right'
        elif direction == 'right' and right < len(filtered_numbers):
            result.append(filtered_numbers[right])
            right += 1
            direction = 'left'
        elif left >= 0:  # Right exhausted
            result.append(filtered_numbers[left])
            left -= 1
        elif right < len(filtered_numbers):  # Left exhausted
            result.append(filtered_numbers[right])
            right += 1
    
    # Pad with last element if needed
    if len(result) < 8 and filtered_numbers:
        result.extend([filtered_numbers[-1]] * (8 - len(result)))
    
    return result

# 映射中间 channel 到输入列表
def map_middle_channels_to_input(channel_list, middle_channels):
    file_path = r"Q:\sachuriga\Sachuriga_Python/quattrocolo-nwb4fp/ASSY-236-F.prb"
    # Read the file and parse the dictionary
    local_vars = {'np': np}
    with open(file_path, 'r') as file:
        exec(file.read(), local_vars)  # Execute the file content with NumPy in scope

        
    channel_groups = local_vars.get('channel_groups')
    if channel_groups is None:
        raise ValueError(f"'channel_groups' not found in {file_path}")

    # Assuming channel_groups is loaded from Step 1
    data = []
    for group_id, group_data in channel_groups.items():
        channels = group_data['channels']
        geometry = group_data['geometry']
        for channel in channels:
            x, y = geometry[channel]
            data.append({
                'group_id': group_id,
                'channel_id': channel,
                'x': x,
                'y': y
            })
    probe_df = pd.DataFrame(data)
    df= probe_df

    output_list = []
    for channel in channel_list:
        group = df[df['channel_id'] == channel]['group_id'].values[0]
        output_list.append(middle_channels[group])
    return output_list


def get_pkl_files(folder_path):
    all_files = os.listdir(folder_path)
    pkl_files = [f for f in all_files if f.endswith("withDLC.pkl")]
    return pkl_files

def find_run_indices(speed_vector, threshold=0.05):
    starts = []
    stops = []
    is_running = False
    for i, speed in enumerate(speed_vector):
        if speed > threshold and not is_running:
            starts.append(i)
            is_running = True
        elif speed <= threshold and is_running:
            stops.append(i - 1)
            is_running = False
    if is_running:  # If still running at the end
        stops.append(len(speed_vector) - 1)
    return starts, stops


def unit_location_ch(file_path:str=r"Q:\sachuriga\Sachuriga_Python/quattrocolo-nwb4fp/ASSY-236-F.prb", x_input: float = 0.0, y_input: float = 0.0):

    # Read the file and parse the dictionary
    with open(file_path, 'r') as file:
        # If the file starts with "channel_groups =", strip it
        content = file.read().replace("channel_groups =", "").strip()
        channel_groups = ast.literal_eval(content)

    # Now channel_groups contains the dictionary

    # Assuming channel_groups is loaded from Step 1
    data = []
    for group_id, group_data in channel_groups.items():
        channels = group_data['channels']
        geometry = group_data['geometry']
        for channel in channels:
            x, y = geometry[channel]
            data.append({
                'group_id': group_id,
                'channel_id': channel,
                'x': x,
                'y': y
            })

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Sort by group_id and channel_id for clarity (optional)
    dataframe = df.sort_values(by=['group_id', 'channel_id']).reset_index(drop=True)

    # Function to find the nearest channel_id given x and y coordinates

        # Calculate Euclidean distance from input (x, y) to all points in the DataFrame
    distances = np.sqrt((dataframe['x'] - x_input)**2 + (dataframe['y'] - y_input)**2)
    # Find the index of the minimum distance
    nearest_idx = distances.idxmin()
    
    # Return the channel_id at that index
    # Example usage with your input
    # x_input = npdata['units']['x'][unit_num]
    # y_input = npdata['units']['y'][unit_num]
    channel_id=dataframe.loc[nearest_idx, 'channel_id']
    distance=distances[nearest_idx]

    return channel_id,distance


def pos2speed(t,x,y,filter_speed=True,min_speed=0.05):
    delta_X = np.diff(x)
    delta_Y = np.diff(y)
    sampling_intervals = np.diff(t)
    average_sampling_interval = np.median(sampling_intervals)
    interval = round(average_sampling_interval, 4)
    samplingrate = 1 / interval

    # Set 250-ms Gaussian std
    std_ms = 250
    std_seconds = std_ms / 1000.0  # convert ms to s
    sigma = std_seconds * samplingrate  # convert std in seconds to std in samples
    truncate = 4.0  # keep same truncate value

    # Calculate distances between points
    delta_S = np.sqrt(delta_X**2 + delta_Y**2)
    speeds = delta_S * samplingrate
    speeds = np.insert(speeds, 0, 0)  # pad to preserve array length

    # Apply Gaussian smoothing
    smoothed_speed = gaussian_filter1d(speeds, sigma=sigma, truncate=truncate)

    if filter_speed == True:
        mask = (speeds >= min_speed)
        filtered_smoothed_speed = smoothed_speed[mask]
        filtered_speeds = speeds[mask]
        valid_mask = mask
    else:
        valid_mask = np.ones_like(speeds, dtype=bool)

    xx = x
    yy = y
    tt = t
    x1 = xx[valid_mask]
    y1 = yy[valid_mask]
    t1 = tt[valid_mask]

    combined_array = np.column_stack((t[valid_mask], x[valid_mask], y[valid_mask]))
    raw_pos = np.column_stack((t, x, y))

    return raw_pos, combined_array, valid_mask, speeds[valid_mask], smoothed_speed, filtered_smoothed_speed

def speed_filtered_spikes(spikes_time, t, mask : list = []):
    # Calculate the extended time bins based on median differences
    median_diff = np.median(np.diff(t[1:]))
    t_ = np.append(t[1:], t[1:][-1] + median_diff)

    # Compute the histogram of spikes across these bins
    spikes_in_bin, _ = np.histogram(spikes_time, t_)

    # If a mask is provided, use it; otherwise return all spikes
    if len(mask) > 0.05:
        spk = spikes_in_bin[mask]
    else:
        spk = spikes_in_bin

    return spk

def Speed_filtered_spikes(spikes_time, t, mask=None):
    """
    Filter original spike times based on a mask (e.g., speed condition).
    Parameters:
    - spikes_time: array of spike timestamps
    - t: array of time points corresponding to the mask
    - mask: boolean array or list of indices where the condition (e.g., running) is true
    
    Returns:
    - filtered_spikes: array of original spike times that fall within masked regions
    """
    if mask is None or len(mask) == 0:
        # If no mask is provided, return all spike times
        return spikes_time
    
    # Ensure mask is a boolean array of the same length as t
    if len(mask) != len(t):
        raise ValueError("Mask length must match time array length")

    # Find the time bins each spike falls into
    spike_indices = np.searchsorted(t, spikes_time, side='right') - 1
    
    # Filter spikes where the mask is True
    valid_spikes = (spike_indices >= 0) & (spike_indices < len(mask))
    filtered_spikes = spikes_time[valid_spikes & mask[spike_indices]]
    
    return filtered_spikes


def load_speed_fromNWB(data):
    scaler = MinMaxScaler()
    # Fit and transform the data
    pos = data
    t = data.as_dataframe().index
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(np.array(pos['x']).reshape(-1, 1)).flatten()
    y = scaler.fit_transform(np.array(pos['y']).reshape(-1, 1)).flatten()

    return np.column_stack((t, x, y))


def load_units_fromNWB(data, unit_num: int = 0):
    units = data
    spikes_time = np.array(units[unit_num].as_series().index)

    return spikes_time

def get_filed_num(matrix):
    # Collect all nonzero values
    distinct_values = set()
    for row in matrix:
        for val in row:
            if val > 0:
                distinct_values.add(val)
    return [v for v in distinct_values]

#def spikes2phase(theta_phase,t,spikestime):

import numpy as np
from scipy import stats

import numpy as np

def calculate_spatial_coherence(place_field_map):
    """
    Calculate spatial coherence of a place field map based on Muller and Kubie (1989).
    Measures first-order spatial autocorrelation without smoothing.
    
    Parameters:
    place_field_map : 2D numpy array
        The place field map with firing rates
    
    Returns:
    float : Spatial coherence value
    """
    # Convert to float and handle potential NaN values
    fmap = np.array(place_field_map, dtype=float)
    if np.all(np.isnan(fmap)):
        return np.nan
        
    # Get dimensions
    rows, cols = fmap.shape
    
    # Create arrays to store values for correlation
    center_values = []
    neighbor_means = []
    
    # Iterate through each pixel (excluding borders)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if not np.isnan(fmap[i, j]):
                # Current pixel value
                center = fmap[i, j]
                
                # Get 8 neighboring pixels
                neighbors = [
                    fmap[i-1, j-1], fmap[i-1, j], fmap[i-1, j+1],
                    fmap[i, j-1],                  fmap[i, j+1],
                    fmap[i+1, j-1], fmap[i+1, j], fmap[i+1, j+1]
                ]
                
                # Calculate mean of valid neighbors
                valid_neighbors = [x for x in neighbors if not np.isnan(x)]
                if valid_neighbors:  # Only if there are valid neighbors
                    neighbor_mean = np.mean(valid_neighbors)
                    center_values.append(center)
                    neighbor_means.append(neighbor_mean)
    
    # Calculate correlation if we have enough points
    if len(center_values) > 1:
        coherence = np.corrcoef(center_values, neighbor_means)[0, 1]
        return coherence if not np.isnan(coherence) else np.nan
    else:
        return np.nan
import numpy as np

def calculate_spatial_coherence1(place_field_map):
    """
    Calculate spatial coherence of a place field map based on Muller and Kubie (1989).
    Measures first-order spatial autocorrelation without smoothing.
    
    Parameters:
    place_field_map : 2D numpy array
        The place field map with firing rates
    
    Returns:
    float : Spatial coherence value
    """
    # Convert to float and handle potential NaN values
    fmap = np.array(place_field_map, dtype=float)
    if np.all(np.isnan(fmap)):
        return np.nan
        
    # Get dimensions
    rows, cols = fmap.shape
    
    # Create arrays to store values for correlation
    center_values = []
    neighbor_means = []
    
    # Iterate through each pixel (excluding borders)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if not np.isnan(fmap[i, j]):
                # Current pixel value
                center = fmap[i, j]
                
                # Get 8 neighboring pixels
                neighbors = [
                    fmap[i-1, j-1], fmap[i-1, j], fmap[i-1, j+1],
                    fmap[i, j-1],                  fmap[i, j+1],
                    fmap[i+1, j-1], fmap[i+1, j], fmap[i+1, j+1]
                ]
                
                # Calculate mean of valid neighbors
                valid_neighbors = [x for x in neighbors if not np.isnan(x)]
                if valid_neighbors:  # Only if there are valid neighbors
                    neighbor_mean = np.mean(valid_neighbors)
                    center_values.append(center)
                    neighbor_means.append(neighbor_mean)
    
    # Calculate correlation if we have enough points
    if len(center_values) > 1:
        coherence = np.corrcoef(center_values, neighbor_means)[0, 1]
        return coherence if not np.isnan(coherence) else np.nan
    else:
        return np.nan
    
def coherence(map_data, normalize='on'):
    """
    Calculate spatial coherence of a rate map based on Muller and Kubie (1989).
    
    Parameters:
    -----------
    map_data : ndarray
        2D rate map, can contain NaNs which will be replaced with 0
    normalize : str, optional
        Whether to normalize the result using arctanh ('on' or 'off', default='on')
    
    Returns:
    --------
    float
        Coherence value
    
    Notes:
    -----
    Uses zero-padding for border values as in the original MATLAB implementation.
    """
    # Input validation
    if not isinstance(map_data, np.ndarray) or len(map_data.shape) != 2 or map_data.size == 0:
        raise ValueError("map_data must be a non-empty 2D numpy array")
    
    if normalize.lower() not in ['on', 'off']:
        raise ValueError("normalize must be 'on' or 'off'")
    
    # Convert to boolean for normalization check
    do_normalization = normalize.lower() == 'on'
    
    # Create the averaging kernel (1/8 for 8 neighbors)
    K = np.array([[0.125, 0.125, 0.125],
                  [0.125, 0.000, 0.125],
                  [0.125, 0.125, 0.125]])
    
    # Handle NaN values by replacing with 0
    map_data = np.nan_to_num(map_data, nan=0.0)
    
    # Perform 2D convolution with 'same' padding (similar to MATLAB's conv2)
    avg_map = signal.convolve2d(map_data, K, mode='same', boundary='fill', fillvalue=0)
    
    # Flatten arrays for correlation (row-major order like MATLAB's reshape with ')
    map_linear = map_data.T.ravel()
    avg_map_linear = avg_map.T.ravel()
    
    # Calculate Pearson correlation
    z, _ = pearsonr(map_linear, avg_map_linear)
    
    # Apply normalization if requested
    if do_normalization:
        z = np.arctanh(z)
    
    return z


def calculate_spatial_stability(map1, map2):
    """
    Calculate spatial stability between two pre-adjusted place field maps 
    using pixel-wise correlation.
    
    Parameters:
    map1 : 2D numpy array of time-adjusted firing rates from first trial
    map2 : 2D numpy array of time-adjusted firing rates from second trial
    
    Returns:
    float : correlation coefficient
    """
    # Ensure maps have same dimensions
    if map1.shape != map2.shape:
        raise ValueError("Map dimensions must match")
    
    # Flatten the maps to 1D arrays
    rates1 = map1.flatten()
    rates2 = map2.flatten()
    
    # Remove any nan values
    valid_pairs = np.logical_and(~np.isnan(rates1), ~np.isnan(rates2))
    valid_rates1 = rates1[valid_pairs]
    valid_rates2 = rates2[valid_pairs]
    
    # Calculate correlation if we have enough valid data points
    if len(valid_rates1) > 1:
        correlation, _ = stats.pearsonr(valid_rates1, valid_rates2)
        return correlation
    return np.nan


def population_vector_correlation(stack1, stack2, full='off', orientation='v'):
    import numpy as np
    """
    Calculates correlation between population vectors.
    
    Parameters:
    -----------
    stack1 : ndarray
        3D matrix representing first population vector (y_bins, x_bins, num_cells)
    stack2 : ndarray
        3D matrix representing second population vector (y_bins, x_bins, num_cells)
    full : str, optional
        'off': returns diagonal elements (2D matrix)
        'on': returns full correlation matrices (3D matrix)
        'vector': returns vector correlation (1D array)
        Default is 'off'
    orientation : str, optional
        'v': vertical (row-wise) processing
        'h': horizontal (column-wise) processing
        Only used when full='vector'. Default is 'v'
    
    Returns:
    --------
    pv_corr : ndarray
        Correlation values (1D, 2D, or 3D array depending on 'full' parameter)
    """
    
    # Define return format constants
    RETURN_2D = 0
    RETURN_3D = 1
    RETURN_1D = 2
    
    # Set default return format and orientation
    return_format = RETURN_2D
    is_orientation_x = False
    
    # Process input parameters
    full = full.lower()
    if full == 'on':
        return_format = RETURN_3D
    elif full == 'vector':
        return_format = RETURN_1D
    
    orientation = orientation.lower()
    if orientation == 'h':
        is_orientation_x = True
    
    if return_format == RETURN_1D and orientation not in ['v', 'h']:
        raise ValueError("Orientation must be 'v' or 'h' when full='vector'")
    
    # Get dimensions
    y_bins1, x_bins1, num_cells1 = stack1.shape
    y_bins2, x_bins2, num_cells2 = stack2.shape
    
    num_cells = min(num_cells1, num_cells2)
    num_x_bins = min(x_bins1, x_bins2)
    num_y_bins = min(y_bins1, y_bins2)
    
    if num_cells1 != num_cells2:
        print(f"Warning: Population vectors have different sizes: {num_cells1} vs {num_cells2}")
    
    # Initialize output array
    if return_format == RETURN_1D:
        pv_corr = np.zeros(num_x_bins if is_orientation_x else num_y_bins)
    elif return_format == RETURN_2D:
        pv_corr = np.zeros((num_y_bins, num_x_bins))
    else:  # RETURN_3D
        pv_corr = np.zeros((x_bins1, x_bins2, num_y_bins))
    
    num_bins = num_x_bins if is_orientation_x else num_y_bins
    
    # Main correlation calculation
    for i in range(num_bins):
        if is_orientation_x:
            stack1d_left = stack1[:, i, :num_cells]  # column
            stack1d_right = stack2[:, i, :num_cells]
        else:
            stack1d_left = stack1[i, :, :num_cells]  # row
            stack1d_right = stack2[i, :, :num_cells]
        
        if return_format == RETURN_1D:
            if is_orientation_x:
                reshaped_left = stack1d_left.reshape(-1)
                reshaped_right = stack1d_right.reshape(-1)
            else:
                reshaped_left = stack1d_left.reshape(-1)
                reshaped_right = stack1d_right.reshape(-1)
        else:
            # Reshape to (num_cells, num_bins) for correlation
            reshaped_left = stack1d_left.T
            reshaped_right = stack1d_right.T
        
        # Calculate correlation
        if return_format == RETURN_3D:
            corr_matrix = np.corrcoef(reshaped_left, reshaped_right)
            # Extract only the relevant part (first half vs second half)
            pv_corr[:, :, i] = corr_matrix[:reshaped_left.shape[1], 
                                         reshaped_left.shape[1]:]
            
        elif return_format == RETURN_2D:
            corr_matrix = np.corrcoef(reshaped_left, reshaped_right)
            diag_vals = np.diag(corr_matrix[:reshaped_left.shape[1], 
                                          reshaped_left.shape[1]:])
            pv_corr[i, :len(diag_vals)] = diag_vals
            pv_corr[i, np.isnan(pv_corr[i])] = 0
            
        else:  # RETURN_1D
            corr_val = np.corrcoef(reshaped_left, reshaped_right)[0, 1]
            pv_corr[i] = corr_val if not np.isnan(corr_val) else 0
    
    return pv_corr


# 1. 带通滤波 (6-11 Hz)
def butter_bandpass(lowcut, highcut, fs, order=4):
    fs = 1250
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)  # 前后向滤波，避免相位移
    return y

def smooth2ripple(data2smooth, fs=1250):
	channels = range(np.shape(data2smooth)[1])
    
	for channel in channels:
		# Since data is in float16 type, we make it smaller to avoid overflows
		# and then we restore it.
		# Mean and std use float64 to have enough space
		# Then we convert the data back to float16
		lfp_filtered = bandpass_filter(data2smooth[:, channel], 160, 225, fs)
		data2smooth[:, channel] = lfp_filtered.astype('float16')
	
	return data2smooth