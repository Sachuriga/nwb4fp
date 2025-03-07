import deeplabcut
import os
from pathlib import Path
import glob

TRACK_METHOD = "skeleton"  
#TRACK_METHOD = "ellipse"
shuffle=1

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
path_config_file = os.path.join(os.getcwd(),r'C:/CR_implant_DLCnet-sachuriga-2023-11-30/config.yaml')
videofile_path = os.path.join(os.getcwd(),r'S:/Sachuriga/Ephys_Vedio/CR_CA1/raw_files')   

# edits = {"engine": "pytorch","batch_size" : 8}
# deeplabcut.auxiliaryfunctions.edit_config(path_config_file, edits)
# print("Start Analyzing the video!")
deeplabcut.analyze_videos(path_config_file,[videofile_path],auto_track=True, videotype='.avi', shuffle=shuffle)

# print("Start convert_detections2tracklets!")
# deeplabcut.convert_detections2tracklets(
#     path_config_file,
#     [videofile_path],
#     videotype='avi',
#     shuffle=shuffle,track_method=TRACK_METHOD
# )

# deeplabcut.stitch_tracklets(
#     path_config_file,
#     [videofile_path],
#     videotype='.avi',
#     shuffle=shuffle,track_method=TRACK_METHOD
# )

deeplabcut.filterpredictions(path_config_file,[videofile_path],shuffle=shuffle,track_method=TRACK_METHOD)
print("Start Creating the video!")
deeplabcut.create_labeled_video(
    path_config_file,
    [videofile_path],
    videotype='.avi',
    shuffle=shuffle,
    color_by="bodypart",
    keypoints_only=False,
    trailpoints=5,
    draw_skeleton=True,
    filtered=True,
    fastmode=True,
    track_method=TRACK_METHOD,
)

