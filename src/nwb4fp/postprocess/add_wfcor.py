import pandas as pd
import numpy as np
from pathlib import Path
import os
def main():
    path = r"S:/Sachuriga/Ephys_Recording/CR_CA1/65410/65410_2023-11-25_13-57-58_A_phy_k_manual"
    add_wf_cor(path)

def add_wf_cor(path):

    path_cluster_group = Path(fr"{path}/cluster_group.tsv")
    path_cluster_metrix = Path(fr"{path}/waveformsfm/extensions/quality_metrics/metrics.csv")
    path_templates = Path(fr"{path}/waveformsfm/extensions/templates_metrics/metrics.csv")
    path_ulocation = Path(fr"{path}/waveformsfm/extensions/unit_locations/unit_locations.npy")
    df0 = pd.read_csv(path_cluster_group, index_col=0, sep='\t')
    df11 = pd.read_csv(path_cluster_metrix)
    df111 = pd.read_csv(path_cluster_metrix)
    
    df1 = pd.merge(df11, df111,left_index=True,right_index=True)
    df2 = pd.DataFrame(np.load(path_ulocation),columns=['x', 'y','z'])
    df3 = pd.merge(df0, df1,left_index=True,right_index=True)
    df4 = pd.merge(df3, df2,left_index=True,right_index=True)

    # List all files in the current directory
    files = os.listdir(path)
    # Filter out files that match the pattern "cluster**.tsv"
    cluster_files = [file for file in files if file.startswith('cluster') and file.endswith('.tsv')]
    # Initialize an empty dataframe to hold the merged data

    merged_df = pd.DataFrame()
    # Loop through the filtered files, read each one, and append to the merged dataframe

    for file in cluster_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, sep='\t')
        merged_df = pd.concat([merged_df,df], axis=1)
        
    df5 =  pd.concat([df4,merged_df], axis=1)
    print(df0)
    df5.to_csv(Path(fr"{path}/cluster_info.tsv"), 
               sep="\t", header=True, 
               index=True)
if __name__== "__main__":
    main()