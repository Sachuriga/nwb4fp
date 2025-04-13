import os 
os.chdir(r'Q:\sachuriga\Sachuriga_Python/quattrocolo-nwb4fp\src')

from neurochat.nc_data import NData
from neurochat.nc_spike import NSpike
from neurochat.nc_spatial import NSpatial
import neurochat.nc_plot as nc_plot
from neurochat.nc_lfp import NLfp
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np
import math
import pynapple as nap
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

import sys
import nwb4fp.analyses.maps as mapp
from nwb4fp.analyses.examples.tracking_plot import plot_ratemap_ax,plot_path
from nwb4fp.analyses.fields import separate_fields_by_laplace, separate_fields_by_dilation,find_peaks,separate_fields_by_laplace_of_gaussian,calculate_field_centers,distance_to_edge_function, remove_fields_by_area, map_pass_to_unit_circle,which_field,compute_crossings
from elephant.statistics import time_histogram, instantaneous_rate
from nwb4fp.analyses import maps
from nwb4fp.analyses.data import pos2speed,speed_filtered_spikes,load_speed_fromNWB,load_units_fromNWB,get_filed_num
from nwb4fp.data.helpers import unit_location_ch
from scipy.ndimage import gaussian_filter
import ast
import pandas as pd


def phase_precession(npdata, unit_num = unit_num):
    ## Load data
    pos_cord = load_speed_fromNWB(npdata['XY_mid_brain'])

    ## filter speed
    raw_pos,combined_array, mask,speeds,smoothed_speed,filtered_speed = pos2speed(pos_cord[:,0], # times
                                pos_cord[:,1], # x
                                pos_cord[:,2], # y
                                filter_speed=True, 
                                min_speed = 0.1)

    ## filter spikes with speed
    unit_num = 21
    raw_pos=raw_pos
    # ## filter spikes with speed
    # spk = speed_filtered_spikes(spikes_time,
    #                             pos_cord[:,0], # times
    #                             mask)
    #for i in range(40):
    spikes_time = load_units_fromNWB(npdata['units'], unit_num = unit_num)
    spk = speed_filtered_spikes(spikes_time,
                                raw_pos[:,0])
    plot_ratemap_ax(raw_pos[:,1], # x
                raw_pos[:,2], # y
                raw_pos[:,0], # times
                spikes_time ,
                box_size=[1.0, 1.0], 
                bin_size=0.05,
                smoothing=0.1)


    x_input = npdata['units']['x'][unit_num]
    y_input = npdata['units']['y'][unit_num]