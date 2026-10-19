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
    parser.add_argument(
        "-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Write this size for every clip of instances.csv instead of reading the videos, for a dataset distributed without them (LSFB: 600 480)",
    )
    args = parser.parse_args()

    if args.size:
        with open(os.path.join(args.data_dir, 'instances.csv'), newline='', encoding='utf-8') as file:
            rows = [(row['id'], *args.size) for row in csv.DictReader(file)]
    else:
        rows = [(os.path.splitext(os.path.basename(path))[0], *video_size(path)) for path in sorted(glob.glob(os.path.join(args.data_dir, 'raw', '*')))]

    with open(os.path.join(args.data_dir, 'metadata', 'video_sizes.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'width', 'height'])
        writer.writerows(rows)

    print('Wrote the size of {} clips'.format(len(rows)))
