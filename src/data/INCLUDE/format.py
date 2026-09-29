import os
import glob
import shutil
import argparse
import polars as pl
from sklearn.model_selection import train_test_split
import json
from extract_keypoints import extract_all, list_videos

parser = argparse.ArgumentParser(description="Convert the extracted INCLUDE dataset into the HandCraft data layout")
parser.add_argument("-data_dir", type=str, default=".", help="INCLUDE dataset root (contains original/)")
parser.add_argument("-model_dir", type=str, default="/disco1/models/mediapipe", help="Directory with the MediaPipe .task models")
parser.add_argument("-test_size", type=float, default=0.3, help="Fraction of videos used for the random test split")
parser.add_argument("-seed", type=int, default=42, help="Random seed for the train/test split")
args = parser.parse_args()

original_dir = os.path.join(args.data_dir, 'original')
raw_dir = os.path.join(args.data_dir, 'raw')
metadata_dir = os.path.join(args.data_dir, 'metadata')

# crear carpetas
os.makedirs(os.path.join(metadata_dir, 'splits'), exist_ok=True)
os.makedirs(raw_dir, exist_ok=True)

# mover archivos a raw y renombrar a <Category>_<sign>#<video>.<ext>
# original/<Category>/<N>. <sign>/[Extra/]<video>.<ext> (some folders have no '<N>. ' prefix)
list_paths = glob.glob(os.path.join(original_dir, '**', '*.MOV'), recursive=True)+glob.glob(os.path.join(original_dir, '**', '*.MP4'), recursive=True)
for path in list_paths:
    split_path = os.path.relpath(path, original_dir).split(os.sep)
    category, sign_dir, video = split_path[0], split_path[1], split_path[-1]
    new_name = category+'_'+sign_dir.split('. ', 1)[-1].replace(" ", "_")+'#'+video
    new_path = os.path.join(raw_dir, new_name)
    if os.path.exists(new_path):
        print('Skipping {}: {} already exists'.format(path, new_path))
        continue
    shutil.move(path, new_path)

# crear instances.csv
# columnas id,sign,signer,start,end
list_paths = list_videos(raw_dir)
instances = {
    'id': [],
    'sign': [],
    'signer': [],
    'start': [],
    'end': []
}
for path in list_paths:
    video_id = os.path.splitext(os.path.basename(path))[0]
    instances['id'].append(video_id)
    instances['sign'].append(video_id[:video_id.find('#')])
    instances['signer'].append("Bender")
    instances['start'].append(0)
    instances['end'].append(1)

df = pl.DataFrame(instances)
df.write_csv(os.path.join(args.data_dir, 'instances.csv'), separator=",")

# crear metadata/sign_to_index.csv
# columnas sign,class
unique_signs = sorted(set(instances['sign']))
sign_to_index = {
    'sign': [],
    'class': []
}
for i, sign in enumerate(unique_signs):
    sign_to_index['sign'].append(sign)
    sign_to_index['class'].append(i)

df = pl.DataFrame(sign_to_index)
df.write_csv(os.path.join(metadata_dir, 'sign_to_index.csv'), separator=",")

# crear metadata/splits/train.json y /metadata/splits/test.json
# lista con nombre de los archivos
# split aleatorio, no el Train_Test_Split oficial (ver README.md)
X = instances['id']
y = instances['sign']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
with open(os.path.join(metadata_dir, 'splits', 'train.json'), "w") as file:
    json.dump(X_train, file)
with open(os.path.join(metadata_dir, 'splits', 'test.json'), "w") as file:
    json.dump(X_test, file)

# extraer poses
extract_all(args.data_dir, args.model_dir)

print('Finished!')
