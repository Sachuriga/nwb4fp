import dlc2kinematics
import matplotlib.pyplot as plt
import numpy as np

file=r"S:/Sachuriga/Ephys_Vedio/CR_CA1/65409_Open_Field_50Hz_A2023-12-05T13_14_39DLC_dlcrnetms5_CR_implant_DLCnetNov30shuffle3_600000_sk.h5"
df, bodyparts, scorer = dlc2kinematics.load_data(file)
df_speed = dlc2kinematics.compute_speed(df,bodyparts=['snout'])
plt.plot(np.float32(df_speed.DLC_dlcrnetms5_CR_implant_DLCnetNov30shuffle3_600000.individual1.snout.speed.tolist())*2500/532)

