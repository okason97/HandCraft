import argparse
import glob
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mediapipe_keypoints import extract_clips


def list_clips(original_dir):
    return sorted(glob.glob(os.path.join(original_dir, 'Signs*', '*', '')))

def clip_id(clip_path):
    return os.path.basename(os.path.normpath(clip_path))

def list_frames(clip_path):
    """
    Colour frames of a clip sorted by frame number (glob returns them in arbitrary order).
    """
    frame_paths = glob.glob(os.path.join(clip_path, '02 Color Frames', '*.jpg'))
    return sorted(frame_paths, key=frame_number)

def frame_number(frame_path):
    return int(os.path.basename(frame_path).split('.')[-2])

def read_times(times_path):
    """
    Read '01 Times/Times.csv' as milliseconds. Each row is
    '<d> days, <h> hours, <m> minutes, <s> seconds, <ms> miliseconds'.
    """
    times = []
    with open(times_path, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            if len(fields) < 5 or not all(field.strip() for field in fields[:5]):
                continue
            days, hours, minutes, seconds, milliseconds = [int(field.split(' ')[-2]) for field in fields[:5]]
            times.append((((days*24+hours)*60+minutes)*60+seconds)*1000+milliseconds)
    return times

def clip_frames(clip_path):
    """
    Yield (frame, timestamp in ms from the first frame) for every colour frame of a clip.
    """
    times = read_times(os.path.join(clip_path, '01 Times', 'Times.csv'))
    # Times.csv can have more rows than there are colour frames
    for frame_path, time in zip(list_frames(clip_path), times, strict=False):
        yield cv2.imread(frame_path), time-times[0]

def extract_all(data_dir, model_dir, workers=1):
    """
    Extract MediaPipe keypoints for every clip in <data_dir>/original and save them
    as <data_dir>/poses/{pose,right_hand,left_hand,face}/<clip_id>.npy
    """
    clips = [(clip_frames, clip_path, clip_id(clip_path))
             for clip_path in list_clips(os.path.join(data_dir, 'original'))]
    extract_clips(clips, data_dir, model_dir, workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints from the DiSPLaY clips in <data_dir>/original")
    parser.add_argument("-data_dir", type=str, default=".", help="DiSPLaY dataset root (contains original/)")
    parser.add_argument("-model_dir", type=str, required=True, help="Directory with the MediaPipe .task models (see download_mediapipe_models.sh)")
    parser.add_argument("-workers", type=int, default=1, help="Number of clips extracted in parallel, one process each")
    args = parser.parse_args()

    extract_all(args.data_dir, args.model_dir, args.workers)

    print('Finished!')
