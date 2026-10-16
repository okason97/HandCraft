import numpy as np
import polars as pl

from generation.autoregressive import write_clip, write_metadata

# HWGAT's 29 keypoints, as in the stgcn/hwgat-data configs
POSES = [["pose", [0, 2, 5, 11, 12, 13, 14, 15, 16]], ["left_hand", [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]], ["right_hand", [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]]]


def box_space_clip(length, seed):
    """
    A clip of the 29 keypoints in box space, shape (length, 58), whose first frame has the nose at (0.5, 1/3) and a
    shoulder width of 1/6, as every clip loaded with norm "box" has.
    """
    rng = np.random.default_rng(seed)
    clip = rng.uniform(0.2, 0.8, (length, 29, 2)).astype(np.float32)
    clip[0, 0] = (0.5, 1 / 3)  # nose
    clip[0, 3] = (0.5 + 1 / 12, 0.45)  # MediaPipe pose keypoint 11, a shoulder
    clip[0, 4] = (0.5 - 1 / 12, 0.45)  # MediaPipe pose keypoint 12, the other shoulder
    return clip.reshape(length, 58)


def make_dataset(root, clips, signs):
    """Writes `clips`, a list of (frames (T, 58), sign), as a dataset at `root`. `signs` are the signs of sign_to_index."""
    sign_to_index = root / "sign_to_index_source.csv"
    pl.DataFrame({"sign": signs, "class": list(range(len(signs)))}).write_csv(sign_to_index)
    for i, (frames, _) in enumerate(clips):
        write_clip(root, i, frames, POSES)
    metadata = {
        "id": [str(i) for i in range(len(clips))],
        "sign": [sign for _, sign in clips],
        "signer": ["Bender"] * len(clips),
        "start": [0] * len(clips),
        "end": [len(frames) for frames, _ in clips],
    }
    write_metadata(root, sign_to_index, metadata, video_sizes=True)
