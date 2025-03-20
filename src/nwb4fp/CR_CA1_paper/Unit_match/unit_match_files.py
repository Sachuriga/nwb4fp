import sys
from pathlib import Path

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import UnitMatchPy.extract_raw_data as erd
import numpy as np 
import probeinterface as pi
import spikeinterface.exporters as sex
from spikeinterface.preprocessing import (bandpass_filter,
                                           common_reference,
                                           whiten)
from spikeinterface.extractors.neoextractors.openephys import OpenEphysBinaryRecordingExtractor
import os
import glob

import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as Overlord
import numpy as np
import matplotlib.pyplot as plt
import UnitMatchPy.GUI as gui
import UnitMatchPy.save_utils as su
import UnitMatchPy.default_params as default_params
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy

def load_filen(animal, day, ephys_path, valid_animals):
    raw_path = []
    phy_folder = []
    
    if animal in valid_animals:
        for suffix in ['A', 'B', 'C']:
            # Raw path pattern: e.g., {ephys_path}\{animal}\{animal}_{day}*{suffix}
            pattern = rf'{ephys_path}\{animal}\{animal}_{day}*{suffix}'
            raw_matches = glob.glob(pattern)
            # Filter and only append if there are valid matches
            raw_filtered = [p for p in raw_matches if os.path.basename(p).endswith(suffix)]
            if raw_filtered:  # Only append if the filtered list is not empty
                raw_path.append(raw_filtered)
            
            # Phy folder pattern: e.g., {ephys_path}\{animal}\{animal}_{day}*A*4match
            pattern = rf'{ephys_path}\{animal}\{animal}_{day}*{suffix}*4match'
            phy_matches = glob.glob(pattern)
            # Filter and only append if there are valid matches
            phy_filtered = [p for p in phy_matches if os.path.basename(p).endswith('4match')]
            if phy_filtered:  # Only append if the filtered list is not empty
                phy_folder.append(phy_filtered)
        
        # Count total number of phy folders across all suffixes
        total_phy_folders = sum(len(sublist) for sublist in phy_folder)
        
        # Return raw_path and phy_folder if 2 or more phy folders exist, else None
        if total_phy_folders >= 2:
            return raw_path, phy_folder
        else:
            return None
    return None  # Return None if animal is not in valid_animals

# ephys_path = 'S:\Sachuriga\Ephys_Recording\CR_CA1'
# raw_path=paths[0][0]
# phy_folder=paths[1][0]
ExtractGoodUnitsOnly = True

def zero_center_waveform(waveform):
    """
    Centers waveform about zero, by subtracting the mean of the first 15 time points.
    This function is useful for Spike Interface where the waveforms are not centered about 0.

    Arguments:
        waveform - ndarray (nUnits, Time Points, Channels, CV)

    Returns:
        Zero centered waveform
    """
    waveform = waveform -  np.broadcast_to(waveform[:,:15,:,:].mean(axis=1)[:, np.newaxis,:,:], waveform.shape)
    return waveform

def run_unitmatch(raw_path,phy_folder,ephys_path,ExtractGoodUnitsOnly:bool=True):
    print(fr"processing the file: {raw_path,phy_folder}")
    Recordings=[]

    for rp in raw_path:
        parts = rp[0].split("\\")
        animal = parts[1]
        day = parts[2].split("_")[1]

        stream_name  = OpenEphysBinaryRecordingExtractor(rp[0],stream_id='0').get_streams(rp[0])[0][0]
        print(fr"Merging step_Before mannual search the stream_name. Auto search result is {stream_name}")
        record_node = stream_name.split("#")[0]
        aquisition_sys = stream_name.split("#")[1]
        recording= se.read_openephys(Path(rp[0]), stream_name=stream_name, load_sync_timestamps=True)

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
        recording_prb = recording.set_probe(probe, group_mode="by_shank")
        Recordings.append(recording_prb)
        
    Sortings = [se.read_phy(Path(file[0])) for file in phy_folder]

    #Will only make average wavefroms for good units
    #Getting good units only
    Sortings[0].get_property_keys() #lists keys for attached propties if 'quality' is not suitbale
    #Good units which will be used in Unit Match
    GoodUnits = []
    UnitsUsed = []
    ids = np.empty(len(Sortings), dtype=object) 
    for i, sorting in enumerate(Sortings):
        UnitIdsTmp = sorting.get_property('original_cluster_id')
        IsGoodTmp = sorting.get_property('quality').astype(dtype='<U50')
        GoodUnits.append(np.stack((UnitIdsTmp,IsGoodTmp), axis = 1))

        UnitsUsed.append(UnitIdsTmp)
        if ExtractGoodUnitsOnly is True:
        # keep = np.argwhere(IsGoodTmp == 'good').squeeze()
            keep = np.argwhere((IsGoodTmp == 'good')).squeeze()
            idss = sorting.get_unit_ids()
            id = idss[keep]
            Sortings[i] = sorting.select_units(id)
            
            ids[i]=id.tolist()
            #ids[i] = Sortings[0].get_property('original_cluster_id')
        else:
                # keep = np.argwhere(IsGoodTmp == 'good').squeeze()
            keep = np.argwhere((IsGoodTmp == 'good') | (IsGoodTmp == 'mua')).squeeze()
            idss = sorting.get_unit_ids()
            id = idss[keep]
            Sortings[i] = sorting.select_units(id)
            
            ids[i]=id.tolist()
            #ids[i] = Sortings[0].get_property('original_cluster_id')

    # # Preprocces the raw data
    for recording in Recordings:
        #recording = spre.phase_shift(recording, inter_sample_shift=None) #correct for time delay between recording channels
        # bad_channel_ids, channel_labels = spre.detect_bad_channels(recording, method="coherence+psd")
        # # remove bad channels
        # recording = recording.remove_channels(bad_channel_ids)
        recording  = spre.bandpass_filter(recording, freq_min=600, freq_max=8000) #highpass
        recording = spre.common_reference(recording=recording, operator="median", reference="global")
        recording = whiten(recording, int_scale=200, mode='local', radius_um=100.0)
        # for motion correction, this can be very slow
        #Uncommented code below to do in session motion correction
        #recording = spre.correct_motion(recording, preset="nonrigid_fast_and_accurate")

    for sorting in Sortings:
        #recording = spre.phase_shift(recording, inter_sample_shift=None) #correct for time delay between recording channels
        sorting.set_property(key='group', values = sorting.get_property("channel_group")) #highpass

    #Split each recording/sorting into 2 halves                    
    for i, sorting in enumerate(Sortings):
        SplitIdx = Recordings[i].get_num_samples() // 2

        SplitSorting = []
        SplitSorting.append(sorting.frame_slice(start_frame=0, end_frame=SplitIdx))
        SplitSorting.append(sorting.frame_slice(start_frame=SplitIdx, end_frame=Recordings[i].get_num_samples()))
        Sortings[i] = SplitSorting 

    for i, recording in enumerate(Recordings):
        SplitIdx = recording.get_num_samples() // 2

        SplitRecording = []
        SplitRecording.append(recording.frame_slice(start_frame=0, end_frame=SplitIdx))
        SplitRecording.append(recording.frame_slice(start_frame=SplitIdx, end_frame=recording.get_num_samples()))

        Recordings[i] = SplitRecording


    #create sorting analyzer for each pair
    Analysers = []
    for i in range(len(Recordings)):
        SplitAnalysers = []
        SplitAnalysers.append(si.create_sorting_analyzer(Sortings[i][0], Recordings[i][0], sparse=False))
        SplitAnalysers.append(si.create_sorting_analyzer(Sortings[i][1], Recordings[i][1], sparse=False))
        Analysers.append(SplitAnalysers)

    #create the fast tempalte extension for each sorting analyser
    AllWaveforms = []
    for i in range(len(Analysers)):
        for half in range(2):
            Analysers[i][half].compute(
                "random_spikes","waveforms",
                method="uniform",
                max_spikes_per_unit=1000)
            Analysers[i][half].compute("noise_levels")
            Analysers[i][half].compute("templates")
            #Analysers[i][half].compute('fast_templates', n_jobs = 0.8,  return_scaled=True)
            #Analysers[i][half].compute('fast_templates', n_jobs = 0.8)
        TemplatesFirst = Analysers[i][0].get_extension('templates')
        TemplatesSecond = Analysers[i][1].get_extension('templates')
        t1 = TemplatesFirst.get_data()
        t2 = TemplatesSecond.get_data()
        AllWaveforms.append(np.stack((t1,t2), axis = -1))

    #Make a channel_postions array
    AllPositions = []
    for i in range(len(Analysers)):
        #postions for first half and second half are the same
        AllPositions.append(Analysers[i][0].get_channel_locations())

    import os
    import shutil
    #UMInputDir = os.path.join(os.getcwd(), 'UMInputData')
    UMInputDir = rf'{ephys_path}\{animal}\{animal}_{day}_UMInputData'

    if os.path.exists(UMInputDir):
        shutil.rmtree(UMInputDir)
        print(f"Delated: {UMInputDir}")
    else:
        print(f"No exist: {UMInputDir}")

    if os.path.exists(UMInputDir):
        shutil.rmtree(UMInputDir)
    os.mkdir(UMInputDir)
    AllSessionPaths = []
    for i in range(len(Recordings)):
        SessionXpath = os.path.join(UMInputDir, f'Session{i+1}') #lets start at 1
        os.mkdir(SessionXpath)

        #save the GoodUnits as a .rsv first column is unit ID,second is 'good' or 'mua'
        GoodUnitsPath = os.path.join(SessionXpath, 'cluster_group.tsv')
        ChannelPositionsPath = os.path.join(SessionXpath, 'channel_positions.npy')
        SaveGoodUnits = np.vstack((np.array(('cluster_id', 'group')), GoodUnits[i])) #Title of colum one is '0000' Not 'cluster_id')
        #SaveGoodUnits[0,0] = 0 # need to be int to use np.savetxt 
        np.savetxt(GoodUnitsPath, SaveGoodUnits, fmt =['%s','%s'], delimiter='\t')
        
        if ExtractGoodUnitsOnly:        
            ids = GoodUnits[i][:,0].tolist()  # Assuming this converts ids to a list
            good = GoodUnits[i][:,1].tolist()  # Assuming this converts good to a list
            temp = [g == 'good' for g in good]  # Creates a boolean list where 'good' is found
            filtered_ids = [[id] for id, t in zip(ids, temp) if t]
            filtered_ids 
            erd.save_avg_waveforms(AllWaveforms[i], SessionXpath, filtered_ids)
        else:
            erd.save_avg_waveforms(AllWaveforms[i], SessionXpath, np.int64(GoodUnits[i][:,0]))
        np.save(ChannelPositionsPath, AllPositions[i])
        AllSessionPaths.append(SessionXpath)

        #get default parameters, can add your own before or after!
    # default of Spikeinterface as by default spike interface extracts waveforms in a different manner.
    param = {'SpikeWidth': 90, 'waveidx': np.arange(15,50), 'PeakLoc': 35}
    param = default_params.get_default_param(param)
    param['no_shanks']=6
    param['shank_dist']=175
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(AllSessionPaths)

    #read in data and select the good units and exact metadata
    #waveform, SessionID, SessionSwitch, WithinSession, GoodUnits, param = util.load_good_waveforms(WavePaths, UnitLabelPaths, param) # 1-step version of above
    #read in data and select the good units and exact metadata

    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths, unit_label_paths,  param, good_units_only =True) 

    #Recenter the waveform at 0, as is not part of spike interface tempaltes 
    waveform = zero_center_waveform(waveform)

    # create clusInfo, contains all unit id/session related info
    clus_info = {'good_units' : GoodUnits, 'session_switch' : session_switch, 'session_id' : session_id, 
                'original_ids' : np.concatenate(GoodUnits) }
    extracted_wave_properties = Overlord.extract_parameters(waveform, channel_pos, clus_info, param)

    #Extract metric scores
    total_score, candidate_pairs, scores_to_include, predictors = Overlord.extract_metric_scores(extracted_wave_properties, session_switch, within_session, param, niter  = 2)

    #Probability analysis
    priorMatch = 1 - (param['n_expected_matches'] / param['n_units']**2 ) # fredom of choose in prior prob?
    Priors = np.array((priorMatch, 1-priorMatch))

    labels = candidate_pairs.astype(int)
    Cond = np.unique(labels)
    ScoreVector = param['score_vector']
    parameter_kernels = np.full((len(ScoreVector), len(scores_to_include), len(Cond)), np.nan)
    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, Cond, param, add_one = 1)
    probability = bf.apply_naive_bayes(parameter_kernels, Priors, predictors, param, Cond)
    output_prob_matrix = probability[:,1].reshape(param['n_units'],param['n_units'])

    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold = 0.75)
    match_threshold = param['match_threshold']
    OutputThreshold = np.zeros_like(output_prob_matrix)
    OutputThreshold[output_prob_matrix > match_threshold] = 1
    plt.imshow(OutputThreshold, cmap = 'Greys')

    amplitude = extracted_wave_properties['amplitude']
    spatial_decay = extracted_wave_properties['spatial_decay']
    avg_centroid = extracted_wave_properties['avg_centroid']
    avg_waveform = extracted_wave_properties['avg_waveform']
    avg_waveform_per_tp = extracted_wave_properties['avg_waveform_per_tp']
    wave_idx = extracted_wave_properties['good_wave_idxs']
    max_site = extracted_wave_properties['max_site']
    max_site_mean = extracted_wave_properties['max_site_mean']
    gui.process_info_for_GUI(output_prob_matrix, match_threshold, scores_to_include, total_score, amplitude, spatial_decay,
                            avg_centroid, avg_waveform, avg_waveform_per_tp, wave_idx, max_site, max_site_mean, 
                            waveform, within_session, channel_pos, clus_info, param)

    matches = np.argwhere(match_threshold == 0.99)
    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)

    SaveDir = rf'{ephys_path}\{animal}\{animal}_{day}_unitmatchResults'

    if os.path.exists(SaveDir):
        shutil.rmtree(SaveDir)
        print(f"Delated: {SaveDir}")
    else:
        print(f"No exist: {SaveDir}")


    su.save_to_output(SaveDir, scores_to_include, matches, output_prob_matrix, avg_centroid, avg_waveform, avg_waveform_per_tp, max_site,
                    total_score, OutputThreshold,  clus_info, param, UIDs=None,matches_curated = True, save_match_table = True,)