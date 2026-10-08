import argparse
import json
import os
import shutil
import urllib.request

import polars as pl

# pinned to the commit the lists were taken from
LISTS_URL = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/98b2db1631356421bf4dfada0aa4716a5ca1ed28/train_test_paths/"

parser = argparse.ArgumentParser(description="Create a data directory with the official INCLUDE train/test split")
parser.add_argument("-data_dir", type=str, default=".", help="INCLUDE dataset root created by format.py")
parser.add_argument("-out_dir", type=str, default="../INCLUDE_official", help="Data directory to create for the official split")
parser.add_argument("-dataset", type=str, default="include", choices=["include", "include50"], help="Official split to use")
args = parser.parse_args()

data_dir = os.path.abspath(args.data_dir)
lists_dir = os.path.join(args.out_dir, 'official_lists')
splits_dir = os.path.join(args.out_dir, 'metadata', 'splits')

# crear carpetas
os.makedirs(lists_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)


def link(src, dst):
    """
    Symlink dst to src. Windows only allows symlinks in developer mode or as administrator:
    there a directory is linked with a junction and a file is copied.
    """
    try:
        os.symlink(src, dst)
    except OSError:
        if os.path.isdir(src):
            import _winapi

            _winapi.CreateJunction(src, dst)
        else:
            shutil.copyfile(src, dst)


# reutilizar poses, instances.csv, sign_to_index.csv y video_sizes.csv (si existe) del dataset original
for name in ['poses', 'instances.csv', os.path.join('metadata', 'sign_to_index.csv'), os.path.join('metadata', 'video_sizes.csv')]:
    src, dst = os.path.join(data_dir, name), os.path.join(args.out_dir, name)
    if os.path.exists(src) and not os.path.lexists(dst):
        link(src, dst)


def read_ids(split):
    """
    Download an official list and convert its paths to the ids used by format.py:
    <Category>/<N>. <sign>/[Extra/]<video>.<ext> -> <Category>_<sign>#<video>
    """
    file_name = '{dataset}_{split}.txt'.format(dataset=args.dataset, split=split)
    list_path = os.path.join(lists_dir, file_name)
    if not os.path.exists(list_path):
        urllib.request.urlretrieve(LISTS_URL + file_name, list_path)
    ids = []
    with open(list_path, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            split_path = line.strip().split('/')
            category, sign_dir, video = split_path[0], split_path[1], split_path[-1]
            ids.append(category + '_' + sign_dir.split('. ', 1)[-1].replace(" ", "_") + '#' + os.path.splitext(video)[0])
    return ids


# solo se pueden usar los videos que tienen poses extraidas
instances = set(pl.read_csv(os.path.join(data_dir, 'instances.csv'))['id'].to_list())
available = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_dir, 'poses', 'pose'))} & instances

# crear metadata/splits/train.json y /metadata/splits/test.json
# train incluye el split de validacion oficial, el entrenamiento separa su propio 10% para validar
splits = {'train': read_ids('train') + read_ids('val'), 'test': read_ids('test')}
for split, ids in splits.items():
    kept = [i for i in ids if i in available]
    print('{split}: {kept} of {total} videos have poses'.format(split=split, kept=len(kept), total=len(ids)))
    with open(os.path.join(splits_dir, split + '.json'), "w") as file:
        json.dump(kept, file)

print('Finished!')
