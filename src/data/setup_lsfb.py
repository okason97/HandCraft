# Download dataset

from lsfb_dataset import Downloader
import os
import numpy as np
import json
import argparse
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

parser = argparse.ArgumentParser(description="A simple argument parser example.")

parser.add_argument("-data_dir", type=str, default="./src/data/", help="Data root directory")

args = parser.parse_args()

lsfb_dir = args.data_dir + 'LSFB'

print('Downloading LSFB')

# downloader = Downloader(dataset='isol', destination=args.data_dir, include_videos=False, timeout=None, check_ssl=False)
downloader = Downloader(dataset='isol', destination=lsfb_dir, include_videos=False, timeout=None)

downloader.download()

print('Download finished')

print('Train data: filter out empty values and samples with more than 60 frames')

root_ids = os.path.join(lsfb_dir, 'metadata', 'splits', "train"+'.json')
with open(root_ids) as f:
    data_ids = json.load(f)
count = 0
for index in data_ids:
    pose_data = np.load(os.path.join(lsfb_dir, 'poses', "pose", index+'.npy'))
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

root_ids = os.path.join(lsfb_dir, 'metadata', 'splits', "test"+'.json')
with open(root_ids) as f:
    data_ids = json.load(f)
count = 0
for index in data_ids:
    pose_data = np.load(os.path.join(lsfb_dir, 'poses', "pose", index+'.npy'))
    if 0 in pose_data.shape or pose_data.shape[0]>60:
        data_ids.remove(index)
        count += 1
print(count)

# Serializing json
json_object = json.dumps(data_ids, indent=4)
 
# Writing to sample.json
with open(root_ids, "w") as outfile:
    outfile.write(json_object)