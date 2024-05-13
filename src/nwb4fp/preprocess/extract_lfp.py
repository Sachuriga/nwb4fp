import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as post
from spikeinterface.preprocessing import (bandpass_filter,
                                           common_reference,resample)
import spikeinterface.qualitymetrics as sqm
from pathlib import Path
import numpy as np
import probeinterface as pi
import math

def main():
    print("main")
    file=r'S:/Sachuriga/Ephys_Recording/CR_CA1/65409/65409_2023-12-04_15-42-35_A'
    extract_lfp(file)

def extract_lfp(file):
    raw_path=file
    #raw_path = r'S:\Sachuriga\Ephys_Recording\CR_CA1\65409\65409_2023-12-04_15-42-35_A'
    stream_name = 'Record Node 101#OE_FPGA_Acquisition_Board-100.Rhythm Data'
    try:
        recording = se.read_openephys(raw_path, stream_name=stream_name, load_sync_timestamps=True)
    except AssertionError:
        try:
            stream_name = 'Record Node 102#OE_FPGA_Acquisition_Board-101.Rhythm Data'
            recording = se.read_openephys(raw_path, stream_name=stream_name, load_sync_timestamps=True)
        except AssertionError:
            stream_name = 'Record Node 101#Acquisition_Board-100.Rhythm Data'
            recording = se.read_openephys(raw_path, stream_name=stream_name, load_sync_timestamps=True)

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

    rec = bandpass_filter(recording, freq_min=1, freq_max=475, dtype='int16')
    rec_car = common_reference(rec, reference='global', operator='average', dtype='int16')
    lfp = resample(rec, 1000, margin_ms=100.0)
    lfp_car = resample(rec_car, 1000, margin_ms=100.0)
    lfp_times = down_sample(recording.get_times(),lfp.get_num_samples())

    lfp= lfp.set_probe(probe, group_mode="by_shank")
    lfp_car= lfp_car.set_probe(probe, group_mode="by_shank")
    path_iron = Path('S:/Sachuriga/Ephys_Recording/CR_CA1/65409/65409_2023-12-04_15-42-35_A_phy_k_manual')

    np.save(path_iron / 'lfp_times.npy', lfp_times)
    lfp1 = lfp.get_traces()
    np.save(path_iron / 'lfp.npy', lfp1) # type: ignore
    lfp1_car = lfp_car.get_traces()
    np.save(path_iron / 'lfp_car.npy', lfp1_car) # type: ignore

def down_sample(msg_cache,msg_n):
    import math

    inc = len(msg_cache) / msg_n  # Calculate increment ratio
    inc_total = 0
    times = np.empty(msg_n)  # Initialize times dictionary
    n = 0
    
    for i in range(msg_n):
        index = math.floor(inc_total)
        if index < len(msg_cache):  # Check to avoid IndexError
            msg_downsampled = msg_cache[index]
            times[i] = msg_downsampled  # Assign to times array
            inc_total += inc
        else:
            break  # Exit loop if index exceeds msg_cache size

    return times

if __name__ == "__main__":
    main()
