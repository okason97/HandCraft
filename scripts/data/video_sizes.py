import argparse
import csv
import glob
import os

import cv2


def video_size(video_path):
    """
    (width, height) of the frames of a video as OpenCV decodes them, which is what the keypoints were extracted from.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError('Could not read a frame of {}'.format(video_path))
    return frame.shape[1], frame.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Write metadata/video_sizes.csv (id,width,height) for the videos in <data_dir>/raw. "
        "DATA.pixel_coords needs it to convert the keypoints from [0, 1] to pixels."
    )
    parser.add_argument("-data_dir", type=str, default=".", help="Dataset root (contains raw/ and metadata/)")
    args = parser.parse_args()

    video_paths = sorted(glob.glob(os.path.join(args.data_dir, 'raw', '*')))
    with open(os.path.join(args.data_dir, 'metadata', 'video_sizes.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'width', 'height'])
        for video_path in video_paths:
            writer.writerow([os.path.splitext(os.path.basename(video_path))[0], *video_size(video_path)])

    print('Wrote the size of {} videos'.format(len(video_paths)))
