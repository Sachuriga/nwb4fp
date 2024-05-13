import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as post
from nwb4fp.postprocess.Get_positions import load_positions,load_positions_h5,test_positions_h5
from nwb4fp.postprocess.get_potential_merge import get_potential_merge
from spikeinterface.preprocessing import (bandpass_filter,
                                           common_reference)
import spikeinterface.exporters as sex
import spikeinterface.qualitymetrics as sqm
from pathlib import Path
import pandas as pd


def main():  
    main()

## Reading the waveform files from phy out_put
def wf4unim(path):
    from pathlib import Path
    import os
    wf = Path(fr"{path}\waveforms_allch\waveforms")
    # List all files in the /mnt/data directory
    file_list = os.listdir(wf)
    # Filter out files that end with '.npy'
    wf_files = [Path(fr"{path}\waveforms_allch\waveforms" + '\\' + file) for file in file_list if file.startswith('waveform') and file.endswith('.npy')]
    index_files = [Path(fr"{path}\waveforms_allch\waveforms" + '\\' + file) for file in file_list if file.startswith('sampled') and file.endswith('.npy')]
    
    return wf_files, index_files
    
def divide_wf(path,sorting):
    import os
    import numpy as np
    spk = pd.DataFrame(sorting.to_spike_vector())
    ids = sorting.get_unit_ids()
    counter=0

    directory = fr"{path}\RawWaveforms"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for id in ids:
        spk_train=spk[spk['unit_index']==counter]['sample_index'].tolist()
        filtered_numbers_f = [index for index,num in enumerate(spk_train) if num < spk_train[-1]/2]
        wf=np.load(fr"{path}\waveforms_allch\waveforms\waveforms_{id}.npy")
        first_half = wf[filtered_numbers_f,:,:]
        print("first" + str(first_half.shape))

        filtered_numbers_s = [index for index,num in enumerate(spk_train) if num > spk_train[-1]/2]
        secound_half = wf[filtered_numbers_s,:,:]
        print("secound" + str(secound_half.shape))
        avg1 = np.mean(first_half, axis=0)
        avg2 = np.mean(secound_half, axis=0)
        
        # Step 2: Combine the two averages
        combined = np.stack((avg1, avg2), axis=-1)
        file_path = os.path.join(directory, f"Unit{counter}_RawSpikes.npy")
        np.save(file_path, combined)
        print(combined.shape)
        counter += 1

if __name__ == "__main__":
    main()