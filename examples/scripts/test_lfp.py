import sys
sys.path.append(r"Q:/sachuriga/Sachuriga_Python/nwb4fprobe/src/")
     
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from neuroconv.datainterfaces import PhySortingInterface
from neuroconv.datainterfaces import OpenEphysRecordingInterface
from neuroconv import ConverterPipe
from nwb4fp.postprocess.Get_positions import load_positions,calc_head_direction,moving_direction,load_positions_h5
from pynwb import NWBHDF5IO, NWBFile
from pynwb import NWBHDF5IO, NWBFile
from dateutil.tz import tzlocal
from nwb4fp.preprocess.down_sample_lfp import down_sample_lfp,add_lfp2nwb

from pynwb.behavior import (
    Position,
    SpatialSeries,
    CompassDirection
)
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.preprocessing import (bandpass_filter,
                                           common_reference,resample)
from pathlib import Path
from preprocess.down_sample import down_sample # type: ignore
import numpy as np
import probeinterface as pi
from pathlib import Path
#from postprocess.Get_positions import load_positions
from pynwb import NWBHDF5IO
from pynwb import NWBHDF5IO
import numpy as np
from pynwb.ecephys import LFP, ElectricalSeries
#from preprocess.down_sample_lfp import down_sample_lfp
import numpy as np
import pandas as pd
import spikeinterface.preprocessing as spre

def lfp2np(raw_path ,file_path):
    raw_path = r'S:\Sachuriga/Ephys_Recording/CR_CA1/65588/65588_2024-03-12_15-17-04_A'
    stream_name = 'Record Node 101#OE_FPGA_Acquisition_Board-100.Rhythm Data'
    try:
        recordingo = se.read_openephys(raw_path, stream_name=stream_name, load_sync_timestamps=True)
    except AssertionError:
        try:
            stream_name = 'Record Node 102#OE_FPGA_Acquisition_Board-101.Rhythm Data'
            recordingo = se.read_openephys(raw_path, stream_name=stream_name, load_sync_timestamps=True)
        except AssertionError:
            stream_name = 'Record Node 101#Acquisition_Board-100.Rhythm Data'
            recordingo = se.read_openephys(raw_path, stream_name=stream_name, load_sync_timestamps=True)


    # from probeinterface import plotting
    manufacturer = 'cambridgeneurotech'
    probe_name = 'ASSY-236-F'
    probe = pi.get_probe(manufacturer, probe_name)
    print(probe)
    # probe.wiring_to_device('cambridgeneurotech_mini-amp-64')
    # map channels to device indices
    mapping_to_device = [
        # connector J2 TOP
        41, 39, 38, 37, 35, 34, 33, 32, 29, 30, 28, 26, 25, 24, 22, 20,
        46, 45, 44, 43, 42, 40, 36, 31, 27, 23, 21, 18, 19, 17, 16, 14,
        # connector J1 BOTTOM
        55, 53, 54, 52, 51, 50, 49, 48, 47, 15, 13, 12, 11, 9, 10, 8,
        63, 62, 61, 60, 59, 58, 57, 56, 7, 6, 5, 4, 3, 2, 1, 0
    ]

    probe.set_device_channel_indices(mapping_to_device)
    probe.to_dataframe(complete=True).loc[:, ["contact_ids", "shank_ids", "device_channel_indices"]]
    probegroup = pi.ProbeGroup()
    probegroup.add_probe(probe)

    pi.write_prb(f"{probe_name}.prb", probegroup, group_mode="by_shank")
    recording_prb = recordingo.set_probe(probe, group_mode="by_shank")
    recp = bandpass_filter(recording_prb, freq_min=1, freq_max=475)
    bad_channel_ids, channel_labels = spre.detect_bad_channels(recp, 
                                                            method='coherence+psd',
                                                            n_neighbors = 9)
    recording_good_channels_f = recp.remove_channels(bad_channel_ids)
    lfp_n50 = spre.notch_filter(recording_good_channels_f, freq=50)
    lfp_n60 = spre.notch_filter(lfp_n50,freq=60)
    rec_lfp_car = common_reference(lfp_n60, reference='global', 
                            operator='median', 
                            dtype='int16')

    lfp_car = resample(rec_lfp_car, resample_rate=1000, margin_ms=100.0)
    lfp = resample(lfp_n60, resample_rate=1000, margin_ms=100.0)
    print(lfp_car.get_channel_ids())

    lfp_times = down_sample(recordingo.get_times(), lfp.get_num_samples())
    file_path = "S:\Sachuriga\Ephys_Recording\CR_CA1/65588/65588_2024-03-12_15-17-04_A_phy_k_manual"
    # Specify the path of the folder
    folder_path = file_path + '/lfp'
    # Check if the folder already exists

    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
    origin = recording_prb.get_property('channel_name')
    new = recording_good_channels_f.get_property('channel_name')
    region = []
    for id in new:
        region.append(np.int32(np.where(origin==id)[0])[0])
    path_iron = Path(file_path+"/lfp")

    for id in new:
        print(id)
        np_lfp_car = load_lfp2mem(lfp_car.channel_slice(channel_ids=[id]))
        np.save(path_iron / fr'lfp_car_{id}.npy',  np_lfp_car.get_traces(return_scaled=True))
        print(fr"complete {id}")
        del np_lfp_car

    for id in new:
        np_lfp = load_lfp2mem(lfp.channel_slice(channel_ids=[id]))
        # np.save(path_iron / fr'lfp_times{id}.npy', lfp_times) # type: ignore
        np.save(path_iron / fr'lfp_raw_{id}.npy', np_lfp.get_traces(return_scaled=True))
        print(fr"complete {id}")
        del np_lfp

def load_lfp2mem(lfp):
    from pathlib import Path
    job_kwargs = dict(n_jobs=40,
                              chunk_duration="5s",
                              progress_bar=True)
    base_folder = Path("C:/temp_lfp")
    preprocessed = "_" + "preprocessed_temp"
    lfp.save(folder=base_folder / preprocessed, overwrite=True, **job_kwargs)
    recording_rec = si.load_extractor(base_folder / preprocessed)
    return recording_rec

if __name__ == "__main__":
    
    lfp2np(raw_path ,file_path)
