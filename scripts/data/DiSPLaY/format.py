import argparse
import json
import os

import polars as pl
from sklearn.model_selection import train_test_split

from extract_keypoints import clip_id, extract_all, frame_number, list_clips, list_frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert the extracted DiSPLaY dataset into the HandCraft data layout")
    parser.add_argument("-data_dir", type=str, default=".", help="DiSPLaY dataset root (contains original/)")
    parser.add_argument("-model_dir", type=str, required=True, help="Directory with the MediaPipe .task models (see download_mediapipe_models.sh)")
    parser.add_argument("-test_size", type=float, default=0.3, help="Fraction of clips used for the random test split")
    parser.add_argument("-seed", type=int, default=42, help="Random seed for the train/test split")
    parser.add_argument("-workers", type=int, default=1, help="Number of clips whose keypoints are extracted in parallel, one process each")
    args = parser.parse_args()

    metadata_dir = os.path.join(args.data_dir, 'metadata')

    # crear carpetas
    print('Creating folders')
    os.makedirs(os.path.join(metadata_dir, 'splits'), exist_ok=True)

    # crear instances.csv
    # columnas id,sign,signer,start,end
    # original/Signs(<a>-<b>)/Sign_<sign>_Performer_<signer>_<repetition>/
    print('Creating metadata')
    instances = {'id': [], 'sign': [], 'signer': [], 'start': [], 'end': []}
    for clip_path in list_clips(os.path.join(args.data_dir, 'original')):
        name = clip_id(clip_path)
        frame_numbers = [frame_number(frame_path) for frame_path in list_frames(clip_path)]

        instances['id'].append(name)
        # signs are written without leading zeros ('01' -> '1'): polars reads the column back as integers
        # when the dataset generation copies sign_to_index.csv, and the names have to stay the same
        instances['sign'].append(str(int(name.split('_')[1])))
        instances['signer'].append(name.split('_')[3])
        instances['start'].append(min(frame_numbers))
        instances['end'].append(max(frame_numbers))

    df = pl.DataFrame(instances)
    df.write_csv(os.path.join(args.data_dir, 'instances.csv'), separator=",")

    # crear metadata/sign_to_index.csv
    # columnas sign,class
    print('Creating sign to class')
    unique_signs = sorted(set(instances['sign']), key=int)
    sign_to_index = {'sign': [], 'class': []}
    for i, sign in enumerate(unique_signs):
        sign_to_index['sign'].append(sign)
        sign_to_index['class'].append(i)

    df = pl.DataFrame(sign_to_index)
    df.write_csv(os.path.join(metadata_dir, 'sign_to_index.csv'), separator=",")

    # crear metadata/splits/train.json y /metadata/splits/test.json
    # lista con nombre de los archivos
    print('Creating train/test')
    X = instances['id']
    y = instances['sign']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    with open(os.path.join(metadata_dir, 'splits', 'train.json'), "w") as file:
        json.dump(X_train, file)
    with open(os.path.join(metadata_dir, 'splits', 'test.json'), "w") as file:
        json.dump(X_test, file)

    # extraer poses
    print('Extracting')
    extract_all(args.data_dir, args.model_dir, args.workers)

    print('Finished!')
