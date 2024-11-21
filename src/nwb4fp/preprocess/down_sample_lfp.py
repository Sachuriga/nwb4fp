import sys
sys.path.append(r'Q:/sachuriga/Sachuriga_Python/quality_metrix')

import spikeinterface.extractors as se
from spikeinterface.preprocessing import (bandpass_filter,
                                           common_reference,resample)
from pathlib import Path
from nwb4fp.preprocess.down_sample import down_sample # type: ignore
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
import spikeinterface as si

def main():
    print(main)

def down_sample_lfp(file_path,raw_path):
    #raw_path = r'S:\Sachuriga/Ephys_Recording/CR_CA1/65409/65409_2023-12-04_15-42-35_A'
    stream_name = 'Record Node 101#OE_FPGA_Acquisition_Board-100.Rhythm Data'
    try:
        recordingo = se.read_openephys(raw_path, stream_name=stream_name, load_sync_timestamps=True)
    except AssertionError:
        try:
            stream_name = 'Record Node 102#OE_FPGA_Acquisition_Board-101.Rhythm Data'
            recordingo = se.read_openephys(raw_path, stream_name=stream_name, load_sync_timestamps=True)
        except AssertionError:
            try:
                stream_name = 'Record Node 102#OE_FPGA_Acquisition_Board-117.Rhythm Data'
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

    rec_lfp_car = common_reference(recording_good_channels_f, reference='global', 
                               operator='median', 
                               dtype='int16')
    
    lfp_car = resample(rec_lfp_car, resample_rate=1000, margin_ms=100.0)
    lfp = resample(recp, resample_rate=1000, margin_ms=100.0)
    print(lfp_car.get_channel_ids())

    lfp_times = down_sample(recordingo.get_times(), lfp.get_num_samples())
    # lfp_car_slice=lfp_car.channel_slice(channel_ids=['CH4', 'CH9', 'CH25',
    #                                                  'CH17', 'CH11', 'CH2',
    #                                                  'CH32', 'CH16', 'CH14',
    #                                                  'CH59', 'CH54', 'CH51',
    #                                                  'CH53', 'CH58', 'CH64',
    #                                                  'CH47', 'CH36', 'CH56'])
    
    # lfp_slice=lfp.channel_slice(channel_ids=['CH4', 'CH9', 'CH25',
    #                                          'CH17', 'CH11', 'CH2',
    #                                          'CH32', 'CH16', 'CH14',
    #                                          'CH59', 'CH54', 'CH51',
    #                                          'CH53', 'CH58', 'CH64',
    #                                          'CH47', 'CH36', 'CH56'])
    origin = recording_prb.get_property('channel_name')
    #new = recording_good_channels_f.get_property('channel_name')
    new = recording_prb.get_property('channel_name')
    region = []
    for id in new:
        region.append(np.int32(np.where(origin==id)[0])[0])

    path_iron = Path(file_path)
    np_lfp_car = load_lfp2mem(lfp_car)
    np.save(path_iron / fr'lfp_car.npy',  np_lfp_car.get_traces(return_scaled=True))
    del np_lfp_car

    np_lfp = load_lfp2mem(lfp)
    np.save(path_iron / fr'lfp_raw.npy',  np_lfp.get_traces(return_scaled=True))
    del np_lfp 

    np.save(path_iron / 'lfp_times.npy', lfp_times) # type: ignore
    # np.save(path_iron / 'lfp_car.npy',  np_lfp_car)
    # np.save(path_iron / 'lfp_raw.npy', np_lfp)  # Save the LFP data
    # print(f"shape {np_lfp.shape}")
    # print(f"descriptions{lfp.get_binary_description()}")
    # print(np_lfp)
    return region

def load_lfp2mem(lfp):
    from pathlib import Path
    job_kwargs = dict(n_jobs=12,
                              chunk_duration="5s",
                              progress_bar=True)
    print("processing lfp data...")
    base_folder = Path("C:/temp_lfp")
    preprocessed = "_" + "preprocessed_temp"
    lfp.save(folder=base_folder / preprocessed, overwrite=True, **job_kwargs)
    # print(fr"LFP shape is {lfp.shape}")
    recording_rec = si.load_extractor(base_folder / preprocessed)
    return recording_rec

def add_lfp2nwb(filename,channel2selec,folder1_path):

    with NWBHDF5IO(filename, "r+") as io:
        read_nwbfile = io.read()
        region=channel2selec
        # create a TimeSeries and add it to the file under the acquisition group
        #device1 = read_nwbfile.add_device(device)
        device1 = read_nwbfile.electrodes.to_dataframe()
        regions=read_nwbfile.create_electrode_table_region(region, "a", name='electrodes')
        # print(device1)
        # print(regions.to_dataframe())

        lfp_times = np.load(fr"{folder1_path}/lfp_times.npy")
        lfp_raw = np.load(fr"{folder1_path}/lfp_raw.npy")
        lfp_car = np.load(fr"{folder1_path}/lfp_car.npy")
        lfp_electrical_series = ElectricalSeries(
            name="lfp_raw",
            data=lfp_raw,
            electrodes=regions,
            starting_time=lfp_times[0],
            rate=1000.0)
        lfp = LFP(electrical_series=lfp_electrical_series)
        ecephys_module = read_nwbfile.create_processing_module(name="lfp_raw", 
                                                        description="1-475Hz, 1000Hz sampling rate, raw extracellular electrophysiology data")
        ecephys_module.add(lfp)
        
        #np_lfp = read_nwbfile.modules["ecephys_raw"].data_interfaces['LFP']['lfp_raw']
        #ch = np_lfp.electrodes.to_dataframe()['channel_name'].tolist()
        #df = pd.DataFrame(np_lfp.data, columns = ch)
        #print(df)
        ecephys_car_module = read_nwbfile.create_processing_module(name="lfp_car", 
                                                        description="1-475Hz, 1000Hz sampling rate, common average reference extracellular electrophysiology data")

        lfp_car_electrical_series = ElectricalSeries(
            name="lfp_car",
            data=lfp_car,
            electrodes=regions,
            starting_time=lfp_times[0],
            rate=1000.0)
        lfp_car = LFP(electrical_series=lfp_car_electrical_series)
        ecephys_car_module.add(lfp_car)
        
        #np_lfp_car = read_nwbfile.modules["ecephys_car"].data_interfaces['LFP']['lfp_car']
        #ch_car = np_lfp_car.electrodes.to_dataframe()['channel_name'].tolist()
        #df = pd.DataFrame(np_lfp.data, columns = ch_car)
        io.write(read_nwbfile)

if __name__== "__main__":
    main()