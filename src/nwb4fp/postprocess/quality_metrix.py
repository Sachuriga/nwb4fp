from pickle import TRUE
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as post
from nwb4fp.postprocess.Get_positions import load_positions,load_positions_h5,test_positions_h5
from nwb4fp.postprocess.get_potential_merge import get_potential_merge
from spikeinterface.preprocessing import (bandpass_filter,
                                           common_reference,
                                           whiten)
import spikeinterface.exporters as sex
import spikeinterface.qualitymetrics as sqm
from pathlib import Path
import pandas as pd
from nwb4fp.postprocess.extract_wf import wf4unim,divide_wf
import spikeinterface.preprocessing as spre
import numpy as np

def main() -> object:
    """
    :rtype: object
    """
    print("main")

def test_clusterInfo(path, temp_folder,save_path_test,vedio_search_directory,idun_vedio_path):
    import shutil
    import probeinterface as pi

    sorting = se.read_phy(folder_path=path, load_all_cluster_properties=True,exclude_cluster_groups = ["noise", "mua"])
    global_job_kwargs = dict(n_jobs=12, chunk_size=10000, chunk_duration="1s", total_memory="32G")
    si.set_global_job_kwargs(**global_job_kwargs)
    temp_path = path.split("_phy")
    raw_path = temp_path[0]
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
    recording_prb = recording.set_probe(probe, group_mode="by_shank")
    rec = bandpass_filter(recording_prb, freq_min=300, freq_max=6000)
    rec_save = common_reference(rec, reference='global', operator='median')

    sorting.set_property(key='group', values = sorting.get_property("channel_group"))
    print(f"Checking the sorting properties")
    
    new_data = pd.DataFrame(columns=['File', 'competability','dlc'])
    temp = path[0 - int(35):]
    path1 = temp.split("/")
    file = path.split("_phy_")
    UD = path1[1].split("_")
    print(file[1])

    try:
        # wf = si.extract_waveforms(rec_save, sorting, folder=fr"{temp_folder}", overwrite=True, 
        #                           sparse=True, method="by_property",by_property="group",max_spikes_per_unit=1000)
        analyzer22 = si.create_sorting_analyzer(sorting=sorting, 
                                               recording=rec_save, 
                                               format='binary_folder', 
                                               folder=fr"{temp_folder}",
                                               overwrite=True)
        try:
            arr_with_new_col,model_num, dlc_path = test_positions_h5(path,vedio_search_directory,raw_path,UD)
            temp_vname = dlc_path.name.split("DLC_dlcrnet")
            vname=temp_vname[0]
            path_ori = dlc_path.parent
            if model_num == 800000:
                new_row = pd.DataFrame({'File': [raw_path], 'competability': "can be merged",'dlc_model': "800000_iteraion"})
                new_row['video_name']= [fr"{vname}.avi"]
                new_row['video_file']= ['file should be there']
            else:  
                new_row = pd.DataFrame({'File': [raw_path], 'competability': "can be merged",'dlc_model': "600000_iteraion"})
                try:
                    shutil.copy2(Path(fr'{path_ori}/{vname}.avi'), Path(fr'{idun_vedio_path}/{vname}.avi'))
                    new_row['video_name']= [fr"{vname}.avi"]
                    new_row['video_file']= ['file transefered']
                except FileNotFoundError:
                    new_row['video_name']= [fr"{vname}.avi"]
                    new_row['video_file']= ['file not exist']
        except IndexError:
            new_row = pd.DataFrame({'File': [raw_path], 'competability': "can be merged",'dlc_model': "file not found"})
            new_row['video_name']= ['please check manualy']
            new_row['video_file']= ['please check manualy']

        print(f"{raw_path} merge complete")
    except AssertionError:
            try:
                arr_with_new_col,model_num, dlc_path = test_positions_h5(path,vedio_search_directory,raw_path,UD)
                temp_vname = dlc_path.name.split("DLC_dlcrnet")
                vname=temp_vname[0]
                path_ori = dlc_path.parent
                if model_num == 800000:
                    new_row = pd.DataFrame({'File': [raw_path], 'competability': "can not be merged",'dlc_model': "800000_iteraion"})
                    new_row['video_name']= [fr"{vname}.avi"]
                    new_row['video_file']= ['file should be there']
                else:  
                    new_row = pd.DataFrame({'File': [raw_path], 'competability': "can not be merged",'dlc_model': "600000_iteraion"})
                    temp_vname = dlc_path.name.split("DLC_dlcrnet")
                    vname=temp_vname[0]
                    path_ori = dlc_path.parent
                    try:
                        shutil.copy2(Path(fr'{path_ori}/{vname}.avi'), Path(fr'{idun_vedio_path}/{vname}.avi'))
                        new_row['video_name']= [fr"{vname}.avi"]
                        new_row['video_file']= ['file transefered']
                    except FileNotFoundError:
                        new_row['video_name']= [fr"{vname}.avi"]
                        new_row['video_file']= ['file not exist']
                        
            except IndexError:
                new_row = pd.DataFrame({'File': [raw_path], 'competability': "can not be merged",'dlc_model': "file not found"})
                new_row['video_name']= ['please check manualy']
                new_row['video_file']= ['please check manualy']

            print(f"{raw_path} no merge")

    existing_data = pd.read_csv(save_path_test)
    # Append the new data to the existing DataFrame
    updated_data = pd.concat([existing_data,new_row], ignore_index=True)
    # Save the updated DataFrame back to a CSV file
    updated_data.to_csv(save_path_test, index=False)

def qualitymetrix(path, temp_folder):

    sorting = se.read_phy(folder_path=path, load_all_cluster_properties=True,exclude_cluster_groups = ["noise", "mua"])
    global_job_kwargs = dict(n_jobs=12, chunk_size=10000, chunk_duration="1s", total_memory="32G")
    si.set_global_job_kwargs(**global_job_kwargs)
    temp_path = path.split("_phy")
    raw_path = temp_path[0]
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

    import probeinterface as pi

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
    recording_prb = recording.set_probe(probe, group_mode="by_shank")
    rec = bandpass_filter(recording_prb, freq_min=300, freq_max=8000)
    bad_channel_ids, channel_labels = spre.detect_bad_channels(rec, method='coherence+psd',n_neighbors = 9)
    recording_good_channels_f = rec.remove_channels(bad_channel_ids)

    rec_save = common_reference(recording_good_channels_f, reference='global', operator='median')
    rec_w = whiten(rec_save, int_scale=200, mode='local', radius_um=100.0)
                   
    sorting.set_property(key='group', values = sorting.get_property("channel_group"))
    print(f"get times for raw sorts{sorting.get_times()}")
    analyzer = si.create_sorting_analyzer(sorting=sorting, recording=rec_w, format='binary_folder', folder=fr"{temp_folder}",overwrite=True)
    we1 = analyzer.compute("random_spikes","waveforms")
    we1 = analyzer.compute("waveforms")
    we1 = analyzer.compute("noise_levels")
    we1 = analyzer.compute("templates")
    #get potential merging sorting objects
    print("processing potential merge...\n")
    
    sort_merge = get_potential_merge(sorting, analyzer)
    sort_merge.set_property(key='group', values = sort_merge.get_property("channel_group"))

    analyzer1 = si.create_sorting_analyzer(sorting=sort_merge, recording=rec_w, format='binary_folder', folder=fr"{temp_folder}_analy",overwrite=True)
    we = analyzer1.compute("random_spikes","waveforms")
    
    we = analyzer1.compute("waveforms")
    we = analyzer1.compute("noise_levels")
    we = analyzer1.compute("templates")
    we = analyzer1.compute("spike_amplitudes")
    we = analyzer1.compute("template_metrics")
    qm = analyzer1.compute("quality_metrics")
    ccg = analyzer1.compute(input="correlograms",
                            window_ms=1000.0,
                            bin_ms=10.0,
                            method="auto")
    
    sim = analyzer1.compute("template_similarity", method="cosine_similarity")
    unit_locations = analyzer1.compute(input="unit_locations", method="monopolar_triangulation")
    qm_ext = analyzer1.compute(input={"principal_components": dict(n_components=10, mode="by_channel_local"),
                                "quality_metrics": dict(skip_pc_metrics=False)})
    qm_data = analyzer1.get_extension("quality_metrics").get_data()
    keep_mask = (qm_data["presence_ratio"] > 0.9) & (qm_data["isi_violations_ratio"] < 0.2) & (np.float64(qm_data["amplitude_median"]) < np.float64(-50.0)) & (qm_data["d_prime"]>4)& (qm_data["l_ratio"]<0.05)
    q = sort_merge.get_property('quality')
    b=q
    b[keep_mask] = 'good'
    b[keep_mask==False]='mua'
    unit_ids=sort_merge.unit_ids
    cluster_group = pd.DataFrame(
            {"cluster_id": [i for i in range(len(unit_ids))], "group": b}
        )
    # print(sort_merge.get_property_keys())
    # print(f"get times for merge sorts{sort_merge.get_times()}")
    # wfm = si.extract_waveforms(rec_save, sort_merge, folder=fr"{temp_folder}", overwrite=True, 
    #                           sparse=True, method="by_property",by_property="group",max_spikes_per_unit=None)
    # post.compute_unit_locations(waveform_extractor=wfm,
    #                             method= 'monopolar_triangulation',
    #                             radius_um=50.)
    # from spikeinterface.postprocessing import compute_principal_components,compute_template_metrics
    # compute_principal_components(waveform_extractor=wfm,n_components=3,whiten=True,mode='by_channel_local',dtype='float64')
    # qm_params = sqm.get_default_qm_params()
    # qm_params["nn_isolation"]["max_spikes"]=10000
    # sqm.compute_quality_metrics(waveform_extractor=wfm, qm_params=qm_params,sparsity=wf.sparsity, skip_pc_metrics=False)
    # print("completet!!!!automerge & quality_metrix_part")
    path_iron = Path(path + "_manual")
    sex.export_to_phy(analyzer1,
                      output_folder = path_iron,
                      remove_if_exists=True,
                      copy_binary=True)
    
    path_iron1 = Path(path + "_4match")
    sex.export_to_phy(analyzer1,
                      output_folder = path_iron1,
                      remove_if_exists=True,
                      copy_binary=False)
    
    cluster_group.to_csv(Path(path_iron1 / r"cluster_group.tsv"), sep="\t", index=False)
    cluster_group.to_csv(Path(path_iron / r"cluster_group.tsv"), sep="\t", index=False)

    phy_TRD = Path(path + "_manual_reports")
    sex.export_report(sorting_analyzer = analyzer1, output_folder=phy_TRD, remove_if_exists=True)
    analyzer1.save_as(folder=Path(path + "_manual/waveformsfm"), format="binary_folder")
    #divide_wf(path_iron,sort_merge)
    unit_groups = sort_merge.get_property("group")
    if unit_groups is None:
        unit_groups = np.zeros(len(unit_ids), dtype="int32")
    channel_group = pd.DataFrame({"cluster_id": [i for i in range(len(unit_ids))], "channel_group": unit_groups})
    channel_group.to_csv(path_iron / "cluster_channel_group.tsv", sep="\t", index=False)
    channel_group.to_csv(path_iron1 / "cluster_channel_group.tsv", sep="\t", index=False)
    print("channel_group")
    print(unit_groups)
    print("completet!!!!_export_to_phy_part")
if __name__ == "__main__":
    main()