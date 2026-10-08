import argparse
import glob
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mediapipe_keypoints import extract_clips, video_frames


def list_videos(raw_dir):
    return sorted(glob.glob(os.path.join(raw_dir, '*.MOV'))+glob.glob(os.path.join(raw_dir, '*.MP4')))

def extract_all(data_dir, model_dir, workers=1):
    """
    Extract MediaPipe keypoints for every video in <data_dir>/raw and save them
    as <data_dir>/poses/{pose,right_hand,left_hand,face}/<video_id>.npy
    """
    clips = [(video_frames, video_path, os.path.splitext(os.path.basename(video_path))[0])
             for video_path in list_videos(os.path.join(data_dir, 'raw'))]
    extract_clips(clips, data_dir, model_dir, workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints from the INCLUDE videos in <data_dir>/raw")
    parser.add_argument("-data_dir", type=str, default=".", help="INCLUDE dataset root (contains raw/)")
    parser.add_argument("-model_dir", type=str, required=True, help="Directory with the MediaPipe .task models (see download_mediapipe_models.sh)")
    parser.add_argument("-workers", type=int, default=1, help="Number of videos extracted in parallel, one process each")
    args = parser.parse_args()

    extract_all(args.data_dir, args.model_dir, args.workers)

    print('Finished!')
