import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    vedio_search_directory = 'S:/Sachuriga/Ephys_Vedio/CR_CA1/'
    folder_path = fr"S:/Sachuriga/Ephys_Recording/CR_CA1/65410/65410_2023-12-04_13-38-02_A/Record Node 102/"
    dlc =  load_positions(path,vedio_search_directory,folder_path,UD)

def test_positions_h5(path,vedio_search_directory,folder_path,UD):
    import glob
    import os
    import pandas as pd
    import numpy as np
    from pathlib import Path

    ''' # Parameters:

        path: This parameter is not used in the function and can be removed.
        vedio_search_directory: This is the directory where the function will look for CSV files containing position data.
        folder_path: This is the directory where the function will look for .npy files containing timestamps and states.
        UD: This is a list or array-like object containing parameters used to construct the search pattern for the CSV files.
        Usage:

        Call the function with the appropriate parameters. For example:
        The function will print the search pattern it uses to find the CSV files.
        The function returns a numpy array containing the position data with timestamps inserted as the first column.
        Output:

        The function returns a numpy array where each row corresponds to a position sample. The first column of the array contains the timestamps, and the remaining columns contain the position data.
        Notes:

        The function assumes a specific directory structure and file naming convention. Make sure your files and directories match these expectations.
        The function only uses the first unique CSV file and the first unique .npy files it finds. If there are multiple matching files, only the first one is used.
        The function extracts specific columns from the CSV file. If your CSV file has a different structure, you may need to modify the column names in the code.
        The function assumes that state 3 in the states.npy file corresponds to the desired timestamps. If your states represent something different, you may need to modify the code.'''
    if path.endswith("phy_k_manual"):
        num2cal = int(41)
    elif path.endswith("phy_k"):
        num2cal = int(35)

    temp = path[0 - num2cal:]
    path1 = temp.split("/")
    UD = path1[1].split("_")

    search_pattern = os.path.join(vedio_search_directory, f"*{UD[0]}*{UD[3]}{UD[1]}*800000_sk_filtered.h5")
    print(search_pattern)
    search_pattern1 = os.path.join(folder_path, '**/**/TTL/timestamps.npy')
    search_pattern3 = os.path.join(folder_path, '**/**/TTL/states.npy')
    #search_pattern2 = os.path.join(folder_path, '**/**/continuous/*/timestamps.npy')

    # Use glob to find files matching the pattern

    matching_files = glob.glob(search_pattern,recursive=True)
    matching_files = np.unique(matching_files)
    print(matching_files)
    try:
        dlc_path=Path(matching_files[0])
        print("Used a 800000 iteration files")
        model_num = 800000
    except IndexError:
        try:
            search_pattern = os.path.join(vedio_search_directory, f"*{UD[0]}*{UD[3]}{UD[1]}*600000_sk_filtered.h5")
            matching_files = glob.glob(search_pattern,recursive=True)
            matching_files = np.unique(matching_files)
            dlc_path=Path(matching_files[0])
            print(f"dlc path: {dlc_path}. Used a 600000 iteration files")
            model_num = 600000
        except IndexError:
            raise IndexError('No file found')
        
    try:
        df = pd.read_hdf(dlc_path, "df_with_missing")
    except KeyError:
        df = pd.read_hdf(dlc_path)   
    bodyparts = df.columns.get_level_values("bodyparts").unique().to_list()
    scorer = df.columns.get_level_values(0)[0]
    
    coords = df[scorer, 'individual1'][[('snout', 'x'), ('snout', 'y'), ('neck', 'x'), ('neck', 'y'), ('back4', 'x'), ('back4', 'y')]]
    positions = np.float32(coords.to_numpy())
    
    print(positions.shape)
    matching_files = glob.glob(search_pattern1,recursive=True)
    matching_files = np.unique(matching_files)
    print(matching_files)
    v_time = np.load(matching_files[0])
    matching_files = glob.glob(search_pattern3,recursive=True)
    matching_files = np.unique(matching_files)
    v_state = np.load(matching_files[0])
    f_time = v_time[np.where(v_state==3)[0]]
    if f_time.shape[0] == 0:
        try:
            f_time = v_time[np.where(v_state==6)[0]]
        except ValueError:
            f_time = v_time[np.where(v_state==3)[0]]
        print('Vedio is 25Hz')

    arr_with_new_col =  np.insert(positions , 0, f_time[:len(positions)], axis=1) # type: ignore
    return arr_with_new_col, model_num, dlc_path

def load_positions_h5(path,vedio_search_directory,folder_path,UD):
    import glob
    import os
    import pandas as pd
    import numpy as np
    from pathlib import Path

    ''' # Parameters:

        path: This parameter is not used in the function and can be removed.
        vedio_search_directory: This is the directory where the function will look for CSV files containing position data.
        folder_path: This is the directory where the function will look for .npy files containing timestamps and states.
        UD: This is a list or array-like object containing parameters used to construct the search pattern for the CSV files.
        Usage:

        Call the function with the appropriate parameters. For example:
        The function will print the search pattern it uses to find the CSV files.
        The function returns a numpy array containing the position data with timestamps inserted as the first column.
        Output:

        The function returns a numpy array where each row corresponds to a position sample. The first column of the array contains the timestamps, and the remaining columns contain the position data.
        Notes:

        The function assumes a specific directory structure and file naming convention. Make sure your files and directories match these expectations.
        The function only uses the first unique CSV file and the first unique .npy files it finds. If there are multiple matching files, only the first one is used.
        The function extracts specific columns from the CSV file. If your CSV file has a different structure, you may need to modify the column names in the code.
        The function assumes that state 3 in the states.npy file corresponds to the desired timestamps. If your states represent something different, you may need to modify the code.'''
    if path.endswith("phy_k_manual"):
        num2cal = int(41)
    elif path.endswith("phy_k"):
        num2cal = int(35)

    temp = path[0 - num2cal:]
    path1 = temp.split("/")
    UD = path1[1].split("_")

    search_pattern = os.path.join(vedio_search_directory, f"*{UD[0]}*{UD[3]}{UD[1]}*800000_sk_filtered.h5")
    print(search_pattern)
    search_pattern1 = os.path.join(folder_path, '**/**/TTL/timestamps.npy')
    search_pattern3 = os.path.join(folder_path, '**/**/TTL/states.npy')
    #search_pattern2 = os.path.join(folder_path, '**/**/continuous/*/timestamps.npy')

    # Use glob to find files matching the pattern

    matching_files = glob.glob(search_pattern,recursive=True)
    matching_files = np.unique(matching_files)
    print(matching_files)
    try:
        dlc_path=Path(matching_files[0])
        print("Used a 800000 iteration files")

    except IndexError:
        try:
            search_pattern = os.path.join(vedio_search_directory, f"*{UD[0]}*{UD[3]}{UD[1]}*600000_sk_filtered.h5")
            matching_files = glob.glob(search_pattern,recursive=True)
            matching_files = np.unique(matching_files)
            dlc_path=Path(matching_files[0])
            print("Used a 600000 iteration files")

        except IndexError:
            raise IndexError('No file found')
        
    try:
        df = pd.read_hdf(dlc_path, "df_with_missing")
    except KeyError:
        df = pd.read_hdf(dlc_path)   
    bodyparts = df.columns.get_level_values("bodyparts").unique().to_list()
    scorer = df.columns.get_level_values(0)[0]
    
    coords = df[scorer, 'individual1'][[('snout', 'x'), ('snout', 'y'), ('neck', 'x'), ('neck', 'y'), ('back4', 'x'), ('back4', 'y')]]
    positions = np.float32(coords.to_numpy())
    
    print(positions.shape)
    matching_files = glob.glob(search_pattern1,recursive=True)
    matching_files = np.unique(matching_files)
    print(matching_files)
    v_time = np.load(matching_files[0])
    matching_files = glob.glob(search_pattern3,recursive=True)
    matching_files = np.unique(matching_files)
    v_state = np.load(matching_files[0])
    f_time = v_time[np.where(v_state==3)[0]]
    if f_time.shape[0] == 0:
        try:
            f_time = v_time[np.where(v_state==6)[0]]
        except ValueError:
            f_time = v_time[np.where(v_state==3)[0]]
        print('Vedio is 25Hz')

    arr_with_new_col =  np.insert(positions , 0, f_time[:len(positions)], axis=1) # type: ignore
    return arr_with_new_col

def load_positions(path,vedio_search_directory,folder_path,UD):

    ''' # Parameters:

        path: This parameter is not used in the function and can be removed.
        vedio_search_directory: This is the directory where the function will look for CSV files containing position data.
        folder_path: This is the directory where the function will look for .npy files containing timestamps and states.
        UD: This is a list or array-like object containing parameters used to construct the search pattern for the CSV files.
        Usage:

        Call the function with the appropriate parameters. For example:
        The function will print the search pattern it uses to find the CSV files.
        The function returns a numpy array containing the position data with timestamps inserted as the first column.
        Output:

        The function returns a numpy array where each row corresponds to a position sample. The first column of the array contains the timestamps, and the remaining columns contain the position data.
        Notes:

        The function assumes a specific directory structure and file naming convention. Make sure your files and directories match these expectations.
        The function only uses the first unique CSV file and the first unique .npy files it finds. If there are multiple matching files, only the first one is used.
        The function extracts specific columns from the CSV file. If your CSV file has a different structure, you may need to modify the column names in the code.
        The function assumes that state 3 in the states.npy file corresponds to the desired timestamps. If your states represent something different, you may need to modify the code.'''
    
    search_pattern = os.path.join(vedio_search_directory, f"*{UD[0]}*{UD[3]}{UD[1]}*600000_sk_filtered.csv")
    print(search_pattern)
    search_pattern1 = os.path.join(folder_path, '**/**/TTL/timestamps.npy')
    search_pattern3 = os.path.join(folder_path, '**/**/TTL/states.npy')
    #search_pattern2 = os.path.join(folder_path, '**/**/continuous/*/timestamps.npy')

    # Use glob to find files matching the pattern

    matching_files = glob.glob(search_pattern,recursive=True)
    matching_files = np.unique(matching_files)

    dlc_path=Path(matching_files[0])

    df = pd.read_csv(dlc_path)
    pos = df[['DLC_dlcrnetms5_CR_implant_DLCnetNov30shuffle3_600000',
                'DLC_dlcrnetms5_CR_implant_DLCnetNov30shuffle3_600000.1',
                'DLC_dlcrnetms5_CR_implant_DLCnetNov30shuffle3_600000.12',
                'DLC_dlcrnetms5_CR_implant_DLCnetNov30shuffle3_600000.13',
                'DLC_dlcrnetms5_CR_implant_DLCnetNov30shuffle3_600000.24',
                'DLC_dlcrnetms5_CR_implant_DLCnetNov30shuffle3_600000.25']]
    
    positions = np.float32(pos[3:].to_numpy())

    matching_files = glob.glob(search_pattern1,recursive=True)
    matching_files = np.unique(matching_files)
    v_time = np.load(matching_files[0])

    matching_files = glob.glob(search_pattern3,recursive=True)
    matching_files = np.unique(matching_files)
    v_state = np.load(matching_files[0])
    f_time = v_time[np.where(v_state==3)[0]]
    if f_time.shape[0] == 0:
        try:
            f_time = v_time[np.where(v_state==6)[0]]
        except ValueError:
            f_time = v_time[np.where(v_state==3)[0]]
        print('Vedio is 25Hz')

    arr_with_new_col =  np.insert(positions , 0, f_time[:len(positions)], axis=1) # type: ignore
    return arr_with_new_col

def calc_head_direction(positions):
    """
    Calculate head direction.

    Calculates the head direction for each position sample pair. Direction
    is defined as east = 0 degrees, north = 90 degrees, west = 180 degrees,
    south = 270 degrees. Direction is set to NaN for missing samples.
    Position matrix contains information about snout and neck. Head
    direction is the counter-clockwise direction from back LED to the front.

    Parameters:
    positions (np.array): Animal's position data, Nx5. Position data should
                          contain timestamps (1 column), X/Y coordinates of
                          first LED (2 and 3 columns correspondingly), X/Y
                          coordinates of the second LED (4 and 5 columns
                          correspondingly).
                          it is assumed that positions[:, 1:2] correspond to
                          front LED, and positions[:, 3:4] to the back LED.
                          The resulting hd is the direction from back LED to
                          the front LED.

    Returns:
    np.array: Vector of head directions in degrees.
    """

    if positions.shape[1] < 5:
        raise ValueError('Position data should be 2D (type ''help calc_head_direction'' for details).')

    x1 = positions[:, 1]
    y1 = positions[:, 2]
    x2 = positions[:, 3]
    y2 = positions[:, 4]

    hd = np.remainder(np.arctan2(y2-y1, x2-x1) * 180 / np.pi + 180, 360)
    return degrees_to_pi_range(hd)

def degrees_to_pi_range(degrees):
    radians = np.radians(degrees)
    return radians

def moving_direction(pos, window_points=[1, 1], step=1):
    newPos = pos.copy()
    nBefore, nAfter = window_points

    if np.isnan(nBefore) or np.isnan(nAfter) or np.isnan(step):
        raise ValueError("Either 'windowPoints' or 'step' contains NaN values. This is not supported")

    num_samples = pos.shape[0]
    mdInd = np.arange(nBefore, num_samples - nAfter, step)

    kernel = np.concatenate([np.ones(nBefore)/nBefore, [0], -np.ones(nAfter)/nAfter])

    dropped_samples = np.setdiff1d(np.arange(num_samples), mdInd)
    newPos[dropped_samples, 1:] = np.nan

    if pos.shape[1] > 3:
        md = np.full((num_samples, 2), np.nan)
        md[mdInd, 0] = calc_direction(pos[:, 1], pos[:, 2], kernel, mdInd)
        md[mdInd, 1] = calc_direction(pos[:, 3], pos[:, 4], kernel, mdInd)
    else:
        md = np.full((num_samples, 1), np.nan)
        md[mdInd, 0] = calc_direction(pos[:, 1], pos[:, 2], kernel, mdInd)

    return degrees_to_pi_range(md), newPos

def calc_direction(x, y, kernel, ind):
    X = np.convolve(x, kernel, 'same')[ind]
    Y = np.convolve(y, kernel, 'same')[ind]
    return np.mod(np.degrees(np.arctan2(Y, X)), 360)

if __name__== "__main__":
    main()