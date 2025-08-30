import dvuploader as dv


# Add file individually

import os

def find_files_ending_with(pattern, root_dir):
    matches = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(pattern):
                matches.append(os.path.join(root, file))
    return matches

# Example usage
root_directory = r'D:/Back_up4_CR_project/NWB files/'  # Replace with your directory
pattern = '_phy_k_manual.zip'
results = find_files_ending_with(pattern, root_directory)
files=[]
for f in results:
    file = dv.File(filepath=fr"{f}")
    print(file)
    files.append(file)
# files = [
#     dv.add_directory(r"D:\Back_up4_CR_project/NWB files"), # Add an entire directory
# ]

DV_URL = "https://dataverse.no/"
API_TOKEN = "4e30feb0-0f56-4e84-b42f-b933e5bb4189"
PID = "https://doi.org/10.18710/RBQOAC"

dvuploader = dv.DVUploader(files=files)
dvuploader.upload(
    api_token=API_TOKEN,
    dataverse_url=DV_URL,
    persistent_id=PID,
    n_parallel_uploads=2, # Whatever your instance can handle
)