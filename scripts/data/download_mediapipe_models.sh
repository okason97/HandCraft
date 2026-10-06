#!/bin/bash
# Usage: ./download_mediapipe_models.sh [out_dir]
# Downloads the MediaPipe models used by the keypoint extraction scripts into out_dir (default: ./mediapipe)
set -e
OUT="${1:-./mediapipe}"
mkdir -p "$OUT"
URL=https://storage.googleapis.com/mediapipe-models

wget -O "$OUT/pose_landmarker_heavy.task" $URL/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
wget -O "$OUT/hand_landmarker.task" $URL/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
wget -O "$OUT/face_landmarker.task" $URL/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# checksums of the models used to extract the keypoints of the published results
md5sum -c <<SUMS
453dec4d02ccc4d3ce812b6de84fa516  $OUT/pose_landmarker_heavy.task
15318430ea3851670fe9914116a9cfad  $OUT/hand_landmarker.task
b0e7274907a1644404fef66b28dd6d85  $OUT/face_landmarker.task
SUMS
