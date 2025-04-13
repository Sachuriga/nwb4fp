import sys
sys.path.append(r'Q:/sachuriga/Sachuriga_Python/quality_metrix')

import spikeinterface.extractors as se
from spikeinterface.preprocessing import (bandpass_filter,
                                           common_reference,resample,notch_filter,zscore,interpolate_bad_channels)
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
from spikeinterface.extractors.neoextractors.openephys import OpenEphysBinaryRecordingExtractor
from spikeinterface.preprocessing import (bandpass_filter,
                                           common_reference,
                                           whiten)
import spikeinterface.exporters as sex
temp_folder=r"C:/temp_lfp"
def main():
    print(main)

def down_sample_lfp(file_path,raw_path):
    GLOBAL_KWARGS = dict(n_jobs=12, total_memory="64G", progress_bar=True, mp_context= "spawn", chunk_size=5000, chunk_duration="1s")
    si.set_global_job_kwargs(**GLOBAL_KWARGS)
    #raw_path = r'S:\Sachuriga/Ephys_Recording/CR_CA1/65409/65409_2023-12-04_15-42-35_A'
    stream_name  = OpenEphysBinaryRecordingExtractor(raw_path,stream_id='0').get_streams(raw_path)[0][0]
    print(fr"Merging step_Before mannual search the stream_name. Auto search result is {stream_name}")
    record_node = stream_name.split("#")[0]
    print(fr"LFP downsampling steps. Auto search result is {stream_name}")
    #stream_name = 'Record Node 101#OE_FPGA_Acquisition_Board-100.Rhythm Data'
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
    bad_channel_ids, channel_labels = spre.detect_bad_channels(recording_prb, 
                                                               method='coherence+psd',
                                                               n_neighbors = 11)
    
    bad_channels=[]
    if any(bad_channel_ids):
        for ch in bad_channel_ids:
            bad_channels.append(np.int64(ch.split("CH")[1])-1)
        path_iron = Path(file_path)
        np.save(path_iron / 'bad_channels.npy', bad_channels) 

    print(rf"bad channels are {bad_channels}")
    print(rf"bad channels labels {channel_labels}")
    
    recording_good_channels_f = recp.remove_channels(bad_channel_ids)
    lfp = resample(recording_prb, resample_rate=1250, margin_ms=100.0)
    #print(lfp_car.get_channel_ids())

    lfp_times = down_sample(recordingo.get_times(), lfp.get_num_samples())

    origin = recording_prb.get_property('channel_names')

    new = recording_prb.get_property('channel_names')
    region = []
    for id in new:
        region.append(np.int32(np.where(origin==id)[0])[0])

    path_iron = Path(file_path)
    np_lfp = load_lfp2mem(lfp)
    np.save(path_iron / fr'lfp_raw.npy',  np_lfp.get_traces(return_scaled=True))
    del np_lfp 

    np.save(path_iron / 'lfp_times.npy', lfp_times) # type: ignore
    
    return region

def down_sample_lfp_test(file_path,raw_path):
    GLOBAL_KWARGS = dict(n_jobs=12, total_memory="64G", progress_bar=True, mp_context= "spawn", chunk_size=5000, chunk_duration="1s")
    si.set_global_job_kwargs(**GLOBAL_KWARGS)
    #raw_path = r'S:\Sachuriga/Ephys_Recording/CR_CA1/65409/65409_2023-12-04_15-42-35_A'
    stream_name  = OpenEphysBinaryRecordingExtractor(raw_path,stream_id='0').get_streams(raw_path)[0][0]
    print(fr"Merging step_Before mannual search the stream_name. Auto search result is {stream_name}")
    record_node = stream_name.split("#")[0]
    print(fr"LFP downsampling steps. Auto search result is {stream_name}")
    #stream_name = 'Record Node 101#OE_FPGA_Acquisition_Board-100.Rhythm Data'
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
    #recp = bandpass_filter(recording_prb, freq_min=1, freq_max=475)
    bad_channel_ids, channel_labels = spre.detect_bad_channels(recording_prb, 
                                                               method='coherence+psd',
                                                               n_neighbors = 11)
    path_iron = Path(file_path)
    bad_channels=[]
    if any(bad_channel_ids):
        for ch in bad_channel_ids:
            bad_channels.append(np.int64(ch.split("CH")[1])-1)
        path_iron = Path(file_path)
        np.save(path_iron / 'bad_channels.npy', bad_channels) 

    print(rf"bad channels are {bad_channels}")
    print(rf"bad channels labels {channel_labels}")

    
    recording_good_channels_f =  interpolate_bad_channels(recording_prb, bad_channel_ids)
    rec_notch = notch_filter(recording_good_channels_f, freq=50, q=100, margin_ms=5.0, dtype=None)
    #rec_comm = common_reference(recording=rec_notch, operator="average", reference="global")
    recp = bandpass_filter(rec_notch, freq_min=1, freq_max=500)
    #zcore_lfp = zscore(recp, mode='mean+std')  

    lfp = resample(recp, resample_rate=1250, margin_ms=100.0)
    #print(lfp_car.get_channel_ids())

    lfp_times = down_sample(recordingo.get_times(), lfp.get_num_samples())
    np.save(path_iron / 'new_channels_name.npy', lfp.get_channel_ids()) 

    print(lfp.get_channel_ids())
    path_iron = Path(file_path)

    np_lfp = load_lfp2mem(lfp)
    np.save(path_iron / fr'lfp_zscore.npy',  np_lfp.get_traces(return_scaled=True))
    del np_lfp 
    lfp_raw =resample(recording_good_channels_f, resample_rate=1250, margin_ms=100.0) 
    np_lfp_raw = load_lfp2mem(lfp_raw)
    np.save(path_iron / fr'lfp_raw.npy', np_lfp_raw .get_traces(return_scaled=True))
    del np_lfp_raw
    np.save(path_iron / 'lfp_times.npy', lfp_times) # type: ignore

    rec = bandpass_filter(recording_prb, freq_min=600, freq_max=8000)
    bad_channel_ids, channel_labels = spre.detect_bad_channels(recording_prb, method='coherence+psd',n_neighbors = 11)
    print(fr"removed {bad_channel_ids}")
    recording_good_ch= rec.remove_channels(bad_channel_ids)
    #recording_good_channels_f = spre.interpolate_bad_channels(rec,bad_channel_ids)

    rec_save = common_reference(recording_good_ch, reference='global', operator='median')
    rec_w = whiten(rec_save, int_scale=200, mode='local', radius_um=100.0)
    memory_size=64
    sorting = se.read_phy(folder_path=file_path, load_all_cluster_properties=True,exclude_cluster_groups = ["noise", "mua"])
    sorting.set_property(key='group', values = sorting.get_property("channel_group"))
    print(f"get times for raw sorts{sorting.get_times()}")
    ## step to analyzer
    GLOBAL_KWARGS = dict(n_jobs=16, total_memory=fr"{memory_size}G", progress_bar=True, mp_context= "spawn", chunk_size=5000, chunk_duration="1s")
    si.set_global_job_kwargs(**GLOBAL_KWARGS)

    analyzer = si.create_sorting_analyzer(sorting=sorting, recording=rec_w, format='memory', folder=fr"{temp_folder}",overwrite=True)
    we1 = analyzer.compute("random_spikes","waveforms")
    we1 = analyzer.compute("noise_levels")
    we1 = analyzer.compute("templates")
    #get potential merging sorting objects
    print("processing potential merge...\n")
    ## Step to creating analyzers
    unit_locations = analyzer.compute(input="unit_locations", method="monopolar_triangulation")
    unit_ids = sorting.unit_ids

    unit_loc_mt = []

    np.save(path_iron / 'unit_loc_ids.npy' ,unit_ids) 
    np.save(path_iron / 'unit_loc_mt.npy' ,unit_locations.data['unit_locations'])
    phy_TRD = Path(file_path + "_manual_reports")
    sex.export_report(sorting_analyzer = analyzer, output_folder=phy_TRD, remove_if_exists=True)

    analyzer1 = si.create_sorting_analyzer(sorting=sorting, recording=rec_w, format='memory', folder=fr"{temp_folder}",overwrite=True)
    analyzer1.compute("random_spikes","waveforms")
    analyzer1.compute("noise_levels")
    analyzer1.compute("templates")
    #get potential merging sorting objects
    print("processing potential merge...\n")
    ## Step to creating analyzers
    unit_locations = analyzer1.compute(input="unit_locations", method="center_of_mass")
    unit_ids = sorting.unit_ids
    unit_loc_cm = []
    np.save(path_iron / 'unit_loc_cm_ids.npy' ,unit_ids) 
    np.save(path_iron / 'unit_loc_cm_mt.npy' ,unit_locations.data['unit_locations'])
    phy_TRD = Path(file_path + "_manual_mass")
    sex.export_report(sorting_analyzer = analyzer1, output_folder=phy_TRD, remove_if_exists=True)

    analyzer2 = si.create_sorting_analyzer(sorting=sorting, recording=rec_w, format='memory', folder=fr"{temp_folder}",overwrite=True)
    analyzer2.compute("random_spikes","waveforms")
    analyzer2.compute("noise_levels")
    analyzer2.compute("templates")
    #get potential merging sorting objects
    print("processing potential merge...\n")
    ## Step to creating analyzers
    unit_locations = analyzer2.compute(input="unit_locations", method="grid_convolution")
    unit_ids = sorting.unit_ids
    unit_loc_gc = []
    np.save(path_iron / 'unit_loc_gc_ids.npy' ,unit_ids) 
    np.save(path_iron / 'unit_loc_gc_mt.npy' ,unit_locations.data['unit_locations'])
    phy_TRD = Path(file_path + "_manual_reports_grid")
    sex.export_report(sorting_analyzer = analyzer2, output_folder=phy_TRD, remove_if_exists=True)

def load_lfp2mem(lfp):
    from pathlib import Path
    GLOBAL_KWARGS = dict(n_jobs=12, total_memory="64G", progress_bar=True, mp_context= "spawn", chunk_size=5000, chunk_duration="1s")
    si.set_global_job_kwargs(**GLOBAL_KWARGS)
    print("processing lfp data...")
    base_folder = Path("C:/temp_lfp")
    preprocessed = "_" + "preprocessed_temp"
    lfp.save(folder=base_folder / preprocessed, overwrite=True)
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