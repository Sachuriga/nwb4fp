import sys
sys.path.append(r'Q:/sachuriga/Sachuriga_Python/quality_metrix')
import spikeinterface as si
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
from spikeinterface.extractors.neoextractors.openephys import OpenEphysBinaryRecordingExtractor

from pynwb.behavior import (
    Position,
    SpatialSeries,
    CompassDirection
)

import numpy as np
def main():
    path = "S:\\Sachuriga/Ephys_Recording/CR_CA1/65409/65409_2023-12-08_16-39-36_A_phy_k_manual"
    sex = "F"
    ages = "P60"
    species = "Mus musculus"
    vedio_search_directory = "S:/Sachuriga/Ephys_Vedio/CR_CA1"
    path_to_save_nwbfile = "S:/Sachuriga/nwb"
    nwbPHYnOPHYS(path, sex, ages, species, vedio_search_directory, path_to_save_nwbfile)  # Added missing argument

def nwbPHYnOPHYS(path,
                 sex,
                 ages,
                 species,
                 vedio_search_directory,
                 path_to_save_nwbfile,
                 skip_qmr: bool = False,
                 post_fix_dlc: str = None):

    if path.endswith("phy_k_manual"):
        num2cal = int(41)
    elif path.endswith("phy_k"):
        num2cal = int(35)
    temp = path[0 - num2cal:]
    path1 = temp.split("/")

    file = path.split("_phy_")
    UD = path1[1].split("_")
    print(file[1])

    GLOBAL_KWARGS = dict(n_jobs=61, total_memory="64G", progress_bar=True, mp_context= "spawn", chunk_size=4000, chunk_duration="1s")
    si.set_global_job_kwargs(**GLOBAL_KWARGS)



    ECEPHY_DATA_PATH = file[0]
    stream_name  = OpenEphysBinaryRecordingExtractor(ECEPHY_DATA_PATH).get_streams(ECEPHY_DATA_PATH)[0][0]
    print(fr"Merging step_Before mannual search the stream_name. Auto search result is {stream_name}")
    record_node = stream_name.split("#")[0]
    aquisition_sys = stream_name.split("#")[1]
    print(fr"Merging step_Before mannual search the record_node. Auto search result is {record_node}")
    #stream_name = 'Record Node 102#OE_FPGA_Acquisition_Board-101.Rhythm Data'
    folder_path = fr"{ECEPHY_DATA_PATH}/{record_node}"

    channel2selec = down_sample_lfp(fr"{path}",fr"{ECEPHY_DATA_PATH}")
    raw_file_folder_path = Path(folder_path)
    try:
        try:
            print("tried" + fr"{stream_name}")
            interface_ophys = OpenEphysRecordingInterface(folder_path=raw_file_folder_path,stream_name=stream_name)
        except AssertionError:
            try:
                folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                print("tried" + fr"{stream_name}")
                stream_name = 'Record Node 102#OE_FPGA_Acquisition_Board-101.Rhythm Data'
                interface_ophys = OpenEphysRecordingInterface(folder_path=folder_path,stream_name=stream_name)
            except AssertionError:
                try:
                    folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 101"
                    stream_name = 'Record Node 101#Acquisition_Board-100.Rhythm Data'
                    print("tried" + fr"{stream_name}")
                    #stream_name = 'Record Node 101#OE_FPGA_Acquisition_Board-101.Rhythm Data'
                    interface_ophys = OpenEphysRecordingInterface(folder_path=folder_path,stream_name=stream_name)
                except ValueError:
                    try:
                        #stream_name = 'Record Node 101#Acquisition_Board-100.Rhythm Data'
                        stream_name = 'Record Node 101#OE_FPGA_Acquisition_Board-100.Rhythm Data'
                        print("tried" + fr"{stream_name}")
                        interface_ophys = OpenEphysRecordingInterface(folder_path=folder_path,stream_name=stream_name)
                    except ValueError:
                        #stream_name = 'Record Node 101#Acquisition_Board-100.Rhythm Data'
                        stream_name = 'Record Node 101#OE_FPGA_Acquisition_Board-101.Rhythm Data'
                        print("tried" + fr"{stream_name}")
                        interface_ophys = OpenEphysRecordingInterface(folder_path=folder_path,stream_name=stream_name)
        except ValueError:
            try:
                folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                stream_name = 'Record Node 102#OE_FPGA_Acquisition_Board-100.Rhythm Data'
                print("tried" + fr"{stream_name}")
                interface_ophys = OpenEphysRecordingInterface(folder_path=raw_file_folder_path,stream_name=stream_name)
            except ValueError:
                folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                stream_name = 'Record Node 102#Acquisition_Board-100.Rhythm Data'
                print("tried" + fr"{stream_name}")
                interface_ophys = OpenEphysRecordingInterface(folder_path=folder_path,stream_name=stream_name)
            # Extract what metadata we can from the source files
        folder1_path = f"{path}"  # Change the folder_path to the location of the data in your system
        print(fr"Animal number is {UD[0]}")
        # if UD[0] == "65410":
        print(fr"Animal number is {UD[0]} replacing the spike times")
        #     sample_num = np.load(fr"{folder1_path}/spike_times.npy")
        #     timestemp = np.load(fr'{folder_path}\experiment1\recording1\continuous\OE_FPGA_Acquisition_Board-101.Rhythm Data/sample_numbers.npy')
        #     print(folder_path)
        #     time_spk = timestemp[sample_num]
        #     np.save(fr"{folder1_path}/spike_times.npy",time_spk)
            
        sample_num = np.load(fr"{folder1_path}/spike_times.npy")
        folder_path = fr"{ECEPHY_DATA_PATH}/{record_node}"
        time_stemp_path = fr"{folder_path}/continuous/{aquisition_sys}/sample_numbers.npy"
        try:
            timestemp = np.load(fr'{time_stemp_path}')
        except FileNotFoundError:
            try:
                folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 101"
                # S:\Sachuriga\Ephys_Recording\CR_CA1\65588\65588_2024-03-06_15-45-53_A\Record Node 101\experiment1\recording1\continuous\Acquisition_Board-100.Rhythm Data
                timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
            except FileNotFoundError:
                try:
                    folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                    timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
                except FileNotFoundError:
                    try:
                        folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                        timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-101.Rhythm Data/sample_numbers.npy')
                    except FileNotFoundError:
                        try:
                            folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                            timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-117.Rhythm Data/sample_numbers.npy')
                        except FileNotFoundError:
                            folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 101"
                            timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
        
        print(folder_path)
        if skip_qmr:
            print("passed replacing the spike times")
            pass
        else:
            time_spk = timestemp[sample_num]
            np.save(fr"{folder1_path}/spike_times.npy",time_spk)
        
        interface_phy = PhySortingInterface(folder_path=folder1_path, verbose=False)
        # For data provenance we add the time zone information to the conversionSS
        converter = ConverterPipe(data_interfaces=[interface_ophys, interface_phy], verbose=False)
        metadata = converter.get_metadata()        
    except FileNotFoundError:
        folder1_path = f"{path}"  # Change the folder_path to the location of the data in your system
        print(fr"Animal number is {UD[0]}")
        # if UD[0] == "65410":
        print(fr"Animal number is {UD[0]} replacing the spike times")
        #     sample_num = np.load(fr"{folder1_path}/spike_times.npy")
        #     timestemp = np.load(fr'{folder_path}\experiment1\recording1\continuous\OE_FPGA_Acquisition_Board-101.Rhythm Data/sample_numbers.npy')
        #     print(folder_path)
        #     time_spk = timestemp[sample_num]
        #     np.save(fr"{folder1_path}/spike_times.npy",time_spk)

        folder_path = fr"{ECEPHY_DATA_PATH}/{record_node}"
        time_stemp_path = fr"{folder_path}/continuous/{aquisition_sys}/sample_numbers.npy"

        sample_num = np.load(fr"{folder1_path}/spike_times.npy")

        print(fr" auto path is here {time_stemp_path}")
        try:
            
            timestemp = np.load(fr'{time_stemp_path}')
        except FileNotFoundError:
            try:
                folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 101"
                # S:\Sachuriga\Ephys_Recording\CR_CA1\65588\65588_2024-03-06_15-45-53_A\Record Node 101\experiment1\recording1\continuous\Acquisition_Board-100.Rhythm Data
                timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
            except FileNotFoundError:
                try:
                    folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                    timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
                except FileNotFoundError:
                    try:
                        folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                        timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-101.Rhythm Data/sample_numbers.npy')
                    except FileNotFoundError:
                        try:
                            folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 102"
                            timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-117.Rhythm Data/sample_numbers.npy')
                        except FileNotFoundError:
                            folder_path = fr"{ECEPHY_DATA_PATH}/Record Node 101"
                            timestemp = np.load(fr'{folder_path}/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
        
        print(folder_path)
        if skip_qmr:
            print("passed replacing the spike times")
            pass
        else:
            time_spk = timestemp[sample_num]
            np.save(fr"{folder1_path}/spike_times.npy",time_spk)
        interface_phy = PhySortingInterface(folder_path=folder1_path, verbose=False)
        # For data provenance we add the time zone information to the conversionSS
        converter = ConverterPipe(data_interfaces=[interface_phy], verbose=False)    
        metadata = converter.get_metadata()
        session_start_time = datetime.now(tzlocal())
        metadata["NWBFile"].update(session_start_time=session_start_time)

    arr_with_new_col = load_positions_h5(path,vedio_search_directory,folder_path,UD,post_fix_dlc=post_fix_dlc)
    # print(f"{arr_with_new_col.shape[1]} output the shape of the array")
    snout2neck = arr_with_new_col[:,[0,1,2,3,4]]
    neck2back4 = arr_with_new_col[:,[0,3,4,5,6]]

    # print(f"snout2neck: {snout2neck}")  
    # print(f"neck2back4: {neck2back4}")

    hd=calc_head_direction(snout2neck)
    bd=calc_head_direction(neck2back4)
    md,new_pos = moving_direction(arr_with_new_col)
    # print(f"position: {new_pos}")
    # print(f"moving Directions: {md.shape}")
    # print(f"Head Directions: {hd}")
    # print(f"Body Directions: {bd}")
    nwbfile = NWBFile(
        session_description="Mouse exploring an open field",  # required
        identifier="sachuriga",  # required
        session_start_time=datetime(2025, 3, 3, 12, tzinfo=ZoneInfo("Norway/Trondheim")))  # required)
    behavior_module = nwbfile.create_processing_module(name="Behavioral data", 
                                                        description="position, head direction, and body direction of the mouse in an open field.")

    position_snout = SpatialSeries(
        name="XY",
        description="Position of snout (x, y) in an open field.",
        data=arr_with_new_col[:,[1,2]],
        timestamps=arr_with_new_col[:,0],
        reference_frame="(0,0) is top left corner")
    position_s = Position(spatial_series = position_snout, name='Snout_position')
    behavior_module.add(position_s)

    position_spatial_series = SpatialSeries(
        name="XY_snout",
        description="Position (x, y) in an open field.",
        data=arr_with_new_col[:,[3,4]],
        timestamps=arr_with_new_col[:,0],
        reference_frame="(0,0) is top left corner")
    position = Position(spatial_series=position_spatial_series, name='Neck_postition')
    behavior_module.add(position)

    hd_direction_spatial_series = SpatialSeries(name="HD",
                                             description="View angle of the subject measured in radians.",
                                             data=hd,
                                             timestamps=snout2neck[:,0],
                                             reference_frame="straight ahead",
                                             unit="radians",)
    bd_direction_spatial_series = SpatialSeries(name="BD",
                                             description="View angle of the subject measured in radians.",
                                             data=bd,
                                             timestamps=neck2back4[:,0],
                                             reference_frame="straight back",
                                             unit="radians",)
    md_direction_spatial_series = SpatialSeries(name="MD",
                                             description="moving angle of the subject measured in radians.",
                                             data=md,
                                             timestamps=neck2back4[:,0],
                                             reference_frame="moving direction",
                                             unit="radians",)   

    lfp_times = np.load(fr"{folder1_path}/lfp_times.npy")

    hd_direction_spatial_series = SpatialSeries(name="lfp_times",
                                             description="lfp_times",
                                             data=lfp_times,
                                             timestamps=lfp_times ,
                                             reference_frame="start of aquisitions",
                                             unit="secounds")

    #ehavior_module = ProcessingModule(name="behavior", description="processed behavioral data")
    
    hd_direction = CompassDirection(spatial_series=hd_direction_spatial_series, name="Head(snout2neck)_Direction")
    bd_direction = CompassDirection(spatial_series=bd_direction_spatial_series, name="Body(neck2back4)_Direction")
    md_direction = CompassDirection(spatial_series=md_direction_spatial_series, name="Moving_Direction")

    behavior_module.add(hd_direction)
    behavior_module.add(bd_direction)
    behavior_module.add(md_direction)
    print(behavior_module)

    #position = Position(spatial_series=position_spatial_series)
    nwbfile_path = fr"{path_to_save_nwbfile}/{path1[1]}.nwb"
    io = NWBHDF5IO(nwbfile_path, mode="w")
    io.write(nwbfile)
    io.close()

    metadata["NWBFile"]["experimenter"] = ["sachuriga,sachuriga"]
    metadata["Subject"] = dict(
        subject_id=UD[0],
        sex=sex,
        age=ages,
        species=species)
    metadata["general_session_id "] = UD[3]
    metadata["lab"] = "quattrocolo lab"
    metadata["institution"] = 'kavili institute for system neuroscience'
    metadata["session_description"] = f"{UD[3]}_room open-field CA1 recording"
    
    print(metadata)
    # For data provenance we add the time zone information to the conversionSS
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
    print("completet!!!!adding conversion to nwb file")
    # channel2selec = [3, 8, 24, 16, 10, 1, 31, 15, 31, 58, 53, 50, 52, 4, 63, 48, 45, 55]
    add_lfp2nwb(nwbfile_path,channel2selec,folder1_path)
    print("completet!!!!adding lfp to nwb file")

def OE_DLC2nwb(path,
                 sex,
                 ages,
                 species,
                 vedio_search_directory,
                 path_to_save_nwbfile,
                 skip_qmr: bool = False,
                 skip_lfp: bool = False,
                 post_fix_dlc: str = None):
    
    global_job_kwargs = dict(n_jobs=32, total_memory="64G",mp_context= "spawn",progress_bar=True)
    si.set_global_job_kwargs(**global_job_kwargs)

    if path.endswith("phy_k_manual"):
        num2cal = int(41)
    elif path.endswith("phy_k"):
        num2cal = int(35)
    temp = path[0 - num2cal:]
    path1 = temp.split("/")

    file = path.split("_phy_")
    UD = path1[1].split("_")
    print(file[1])

    ECEPHY_DATA_PATH = file[0]
    stream_name  = OpenEphysBinaryRecordingExtractor(ECEPHY_DATA_PATH,stream_id='0').get_streams(ECEPHY_DATA_PATH)[0][0]
    print(fr"Merging step_Before mannual search the stream_name. Auto search result is {stream_name}")
    record_node = stream_name.split("#")[0]
    aquisition_sys = stream_name.split("#")[1]
    print(fr"Merging step_Before mannual search the record_node. Auto search result is {record_node}")
    #stream_name = 'Record Node 102#OE_FPGA_Acquisition_Board-101.Rhythm Data'
    folder_path = fr"{ECEPHY_DATA_PATH}/{record_node}"
    if skip_lfp == True:
        pass
    else:
        channel2selec = down_sample_lfp(fr"{path}",fr"{ECEPHY_DATA_PATH}")

    raw_file_folder_path = Path(folder_path)
    interface_ophys = OpenEphysRecordingInterface(folder_path=raw_file_folder_path,stream_name=stream_name)

    folder1_path = f"{path}"
    sample_num = np.load(fr"{folder1_path}/spike_times.npy")
    folder_path = fr"{ECEPHY_DATA_PATH}/{record_node}"
    print(fr"searching for the sample numbers: auto match--{folder_path}")
    time_stemp_path = fr"{folder_path}/experiment1/recording1/continuous/{aquisition_sys}/sample_numbers.npy"
    timestemp = np.load(Path(fr'{time_stemp_path}'))

    if skip_qmr:
        print("passed replacing the spike times")
        pass
    else:
        time_spk = timestemp[sample_num]
        np.save(fr"{folder1_path}/spike_times.npy",time_spk)
    
    interface_phy = PhySortingInterface(folder_path=folder1_path, verbose=False)
    # For data provenance we add the time zone information to the conversionSS
    converter = ConverterPipe(data_interfaces=[interface_ophys, interface_phy], verbose=False)
    metadata = converter.get_metadata()        
  

    arr_with_new_col, full_cords_arr = load_positions_h5(path,vedio_search_directory,folder_path,UD,post_fix_dlc=post_fix_dlc)
    # print(f"{arr_with_new_col.shape[1]} output the shape of the array")
    snout2neck = arr_with_new_col[:,[0,1,2,3,4]]
    neck2back4 = arr_with_new_col[:,[0,3,4,5,6]]

    # print(f"snout2neck: {snout2neck}")  
    # print(f"neck2back4: {neck2back4}")

    hd=calc_head_direction(snout2neck)
    bd=calc_head_direction(neck2back4)
    md,new_pos = moving_direction(arr_with_new_col)
    # print(f"position: {new_pos}")
    # print(f"moving Directions: {md.shape}")
    # print(f"Head Directions: {hd}")
    # print(f"Body Directions: {bd}")
    nwbfile = NWBFile(
        session_description="Open filed exploration with CA1 neurons recording with 6 shank f probe from cambrage neuro tech,",  # required
        identifier="sachuriga",  # required
        session_start_time=datetime(2025, 3, 3, 12, tzinfo=ZoneInfo("Europe/Oslo")))  # required)
    behavior_module = nwbfile.create_processing_module(name="Behavioral data", 
                                                        description="position, head direction, and body direction of the mouse in an open field.")

    position_snout = SpatialSeries(
        name="XY",
        description="Position of snout (x, y) in an open field.",
        data=arr_with_new_col[:,[1,2]],
        timestamps=arr_with_new_col[:,0],
        reference_frame="(0,0) is top left corner")
    position_s = Position(spatial_series = position_snout, name='Snout_position')
    behavior_module.add(position_s)

    position_spatial_series = SpatialSeries(
        name="XY_mid_brain",
        description="Position (x, y) in an open field.",
        data=arr_with_new_col[:,[3,4]],
        timestamps=arr_with_new_col[:,0],
        reference_frame="(0,0) is top left corner")
    position = Position(spatial_series=position_spatial_series, name='Neck_postition')
    behavior_module.add(position)

    hd_direction_spatial_series = SpatialSeries(name="HD",
                                             description="View angle of the subject measured in radians.",
                                             data=hd,
                                             timestamps=snout2neck[:,0],
                                             reference_frame="straight ahead",
                                             unit="radians",)
    bd_direction_spatial_series = SpatialSeries(name="BD",
                                             description="View angle of the subject measured in radians.",
                                             data=bd,
                                             timestamps=neck2back4[:,0],
                                             reference_frame="straight back",
                                             unit="radians",)
    md_direction_spatial_series = SpatialSeries(name="MD",
                                             description="moving angle of the subject measured in radians.",
                                             data=md,
                                             timestamps=neck2back4[:,0],
                                             reference_frame="moving direction",
                                             unit="radians",)   

    lfp_times = np.load(fr"{folder1_path}/lfp_times.npy")

    hd_direction_spatial_series = SpatialSeries(name="lfp_times",
                                             description="lfp_times",
                                             data=lfp_times,
                                             timestamps=lfp_times ,
                                             reference_frame="start of aquisitions",
                                             unit="secounds")

    #ehavior_module = ProcessingModule(name="behavior", description="processed behavioral data")
    body_parts = ["Snout","Left_ear","Right_ear","Mid_brain","Neck","Back1","Back2","Back3","Back4","Tail1","Tail2","Tail3","Tail4"]
    for i in range(13):
        col1 = i*2 
        col2 = i*2 +2 
        full_cord_spatial_series = SpatialSeries(name=fr"{body_parts[i]}_cords",
                                                description=fr"X,Y cordinations for the {body_parts[i]}",
                                                data=full_cords_arr[:,col1:col2],
                                                timestamps=snout2neck[:,0],
                                                reference_frame="straight ahead",
                                                unit="pixels in the farme")
        full_cords_dlc = Position(spatial_series=full_cord_spatial_series, name=fr"{body_parts[i]}_cords")
        behavior_module.add(full_cords_dlc)

    hd_direction = CompassDirection(spatial_series=hd_direction_spatial_series, name="Head(snout2neck)_Direction")
    bd_direction = CompassDirection(spatial_series=bd_direction_spatial_series, name="Body(neck2back4)_Direction")
    md_direction = CompassDirection(spatial_series=md_direction_spatial_series, name="Moving_Direction")

    behavior_module.add(hd_direction)
    behavior_module.add(bd_direction)
    behavior_module.add(md_direction)
    print(behavior_module)

    #position = Position(spatial_series=position_spatial_series)
    nwbfile_path = fr"{path_to_save_nwbfile}/{path1[1]}.nwb"
    io = NWBHDF5IO(nwbfile_path, mode="w")
    io.write(nwbfile)
    io.close()

    metadata["NWBFile"]["experimenter"] = ["sachuriga,sachuriga"]
    metadata["Subject"] = dict(
        subject_id=UD[0],
        sex=sex,
        age=ages,
        species=species)
    metadata["general_session_id "] = UD[3]
    metadata["lab"] = "Quattrocolo lab"
    metadata["institution"] = 'Kavili institute for system neuroscience'
    metadata["session_description"] = f"{UD[3]}_room open-field CA1 recording"
    
    


    print(metadata)
    # For data provenance we add the time zone information to the conversionSS
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
    print("completet!!!!adding conversion to nwb file")
    # channel2selec = [3, 8, 24, 16, 10, 1, 31, 15, 31, 58, 53, 50, 52, 4, 63, 48, 45, 55]
    add_lfp2nwb(nwbfile_path,channel2selec,folder1_path)
    print("completet!!!!adding lfp to nwb file")

if __name__== "__main__":
    main()