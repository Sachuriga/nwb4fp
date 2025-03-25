import numpy as np
from scipy.ndimage import gaussian_filter
import seaborn as sns
from itertools import chain
import pandas as pd
import ast

def df2results(df):
    # Assuming your data is in arrays x, y, and speed
    # If they're in a pandas DataFrame called df, you could extract them like:
    # x = df['x'].values
    # y = df['y'].values
    # speed = df['speed'].values

    # Create 2D histogram with 200x200 bins
    bins = 200
    x_bins = np.linspace(0, 1, bins + 1)  # +1 because these are bin edges
    y_bins = np.linspace(0, 1, bins + 1)

    x_control = list(chain.from_iterable(df['x'])) 
    y_control = list(chain.from_iterable(df['y'])) 
    speeds = list(chain.from_iterable(df['smoothed_speed'])) 


    # Calculate the 2D histogram of counts
    hist_counts, x_edges, y_edges = np.histogram2d(x_control, y_control, bins=[x_bins, y_bins])

    # Calculate the sum of speeds in each bin
    hist_speed_sum, _, _ = np.histogram2d(x_control, y_control, bins=[x_bins, y_bins], weights=speeds)

    # Avoid division by zero by using where
    average_speed = np.divide(hist_speed_sum, hist_counts, 
                            out=np.zeros_like(hist_speed_sum), 
                            where=hist_counts != 0)

    # Get bin centers for x and y (instead of edges)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Now you have:
    # X: 200x200 array of x coordinates
    # Y: 200x200 array of y coordinates
    # average_speed: 200x200 array of averaged speeds
    smoothed_speed = gaussian_filter(average_speed*50, sigma=20/2) 
    data_speedss = pd.DataFrame(smoothed_speed, index=y_centers, columns=x_centers)

    return data_speedss


# Assuming files_df contains h5py objects in a column, e.g., 'unit_tables'
def convert_h5py_to_picklable(df, column_name):
    # Convert h5py objects to NumPy arrays or other picklable types
    df[column_name] = df[column_name].apply(lambda x: x[()] if isinstance(x, h5py.Dataset) else x)
    return df

def df2results_sns(df):
    x_control = list(chain.from_iterable(df['x'])) 
    y_control = list(chain.from_iterable(df['y'])) 


    sample_size = min(10000, len(x_control))  # Don’t exceed original size
    indices = np.random.choice(len(x_control), sample_size, replace=False)

    x_sampled = [x_control[i] for i in indices]
    y_sampled = [y_control[i] for i in indices]

    # Create sampled DataFrame
    data = pd.DataFrame({'x': x_sampled, 'y': y_sampled})
    return data

def unit_location_ch(file_path:str=r"Q:\sachuriga\Sachuriga_Python/quattrocolo-nwb4fp/ASSY-236-F.prb", x: float = 0.0, y: float = 0.0):

    import pandas as pd
    x_input=x
    y_input=y

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

    # Create a DataFrame
    df = pd.DataFrame(data)
    # Sort by group_id and channel_id for clarity (optional)
    dataframe = df.sort_values(by=['group_id', 'channel_id']).reset_index(drop=True)

    # Function to find the nearest channel_id given x and y coordinates
        # Calculate Euclidean distance from input (x, y) to all points in the DataFrame
    distances = np.sqrt((dataframe['x'] - x_input)**2 + (dataframe['y'] - y_input)**2)
    nearest_idx = distances.idxmin()

    # Return the channel_id at that index
    channel_id=dataframe.loc[nearest_idx, 'channel_id']

    return channel_id