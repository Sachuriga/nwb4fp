# Author: Sachuriga
# Email: sachuriga3@gmail.com
# Purpose: Process CSV files and add their columns to corresponding NWB files

import pandas as pd
from pynwb import NWBHDF5IO
import os

# Set your base folder path
base_folder = 'S:/Sachuriga/nwb/test4neo'

# Get all files in the directory
all_files = os.listdir(base_folder)

# Filter for CSV and NWB files and create pairs
csv_files = [f for f in all_files if f.endswith('.csv')]
nwb_files = [f for f in all_files if f.endswith('.nwb')]

# Process each pair of files
for csv_file in csv_files:
    # Try to find a matching NWB file (assuming similar base names)
    base_name = os.path.splitext(csv_file)[0]
    nwb_file = f"{base_name}.nwb"
    
    if nwb_file in nwb_files:
        csvpath = os.path.join(base_folder, csv_file)
        nwbpath = os.path.join(base_folder, nwb_file)
        
        try:
            # Read the CSV file
            df = pd.read_csv(csvpath)
            
            # Get column names
            column_names = df.columns.tolist()
            
            # Open and modify NWB file
            with NWBHDF5IO(nwbpath, "a") as io:
                nwb = io.read()
                for col in column_names:
                    print(f"Processing column: {col}")
                    nwb.units.add_column(
                        name=f"matlab_{col}",
                        description=f"values of {col}",
                        data=df[col].tolist()
                    )
                
                # Uncomment these if you need to modify subject info
                # nwb.subject.genotype = "Sst-IRES-Cre"
                # nwb.subject.set_modified()
                
                io.write(nwb)
            print(f"Successfully processed {csv_file} into {nwb_file}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    else:
        print(f"Warning: No matching NWB file found for {csv_file}")

print("Processing complete!")