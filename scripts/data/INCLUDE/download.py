import argparse
import json
import os

import wget

parser = argparse.ArgumentParser(description="Download the INCLUDE dataset from Zenodo (record 4010759)")
parser.add_argument("-files_json", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files.json'), help="Zenodo file listing")
parser.add_argument("-out_dir", type=str, default=".", help="Download directory")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

with open(args.files_json, 'r') as file:
    data = json.load(file)

for entry in data['entries']:
    url = entry['links']['self'].replace('api/', '') + '?download=1'
    filename = wget.download(url, out=args.out_dir)
