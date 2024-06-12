# Download dataset

from lsfb_dataset import Downloader
import os
import numpy as np
import json

print('Downloading LSFB')

data_dir = "./src/data/"

downloader = Downloader(dataset='isol', destination=data_dir, include_videos=False, timeout=120)

downloader.download()

data_dir = data_dir + 'LSFB'

print('Download finished')

print('Train data: filter out empty values and samples with more than 60 frames')

root_ids = os.path.join(data_dir, 'metadata', 'splits', "train"+'.json')
with open(root_ids) as f:
    data_ids = json.load(f)
count = 0
for index in data_ids:
    pose_data = np.load(os.path.join(data_dir, 'poses', "pose", index+'.npy'))
    if 0 in pose_data.shape or pose_data.shape[0]>60:
        data_ids.remove(index)
        count += 1
print(count)

# Serializing json
json_object = json.dumps(data_ids, indent=4)
 
# Writing to sample.json
with open(root_ids, "w") as outfile:
    outfile.write(json_object)

print('Test data: filter out empty values and samples with more than 60 frames')

root_ids = os.path.join(data_dir, 'metadata', 'splits', "test"+'.json')
with open(root_ids) as f:
    data_ids = json.load(f)
count = 0
for index in data_ids:
    pose_data = np.load(os.path.join(data_dir, 'poses', "pose", index+'.npy'))
    if 0 in pose_data.shape or pose_data.shape[0]>60:
        data_ids.remove(index)
        count += 1
print(count)

# Serializing json
json_object = json.dumps(data_ids, indent=4)
 
# Writing to sample.json
with open(root_ids, "w") as outfile:
    outfile.write(json_object)