import multiprocessing
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

POSE_PARTS = ['pose', 'right_hand', 'left_hand', 'face']


def center_region(image, landmarks, padding=50):
    if not landmarks:
        return image, (0, 0)

    x_coords = [int(landmark.x * image.shape[1]) for landmark in landmarks]
    y_coords = [int(landmark.y * image.shape[0]) for landmark in landmarks]

    left = max(0, min(x_coords) - padding)
    top = max(0, min(y_coords) - padding)
    right = min(image.shape[1], max(x_coords) + padding)
    bottom = min(image.shape[0], max(y_coords) + padding)

    centered = image[top:bottom, left:right].copy()
    return centered, (left, top)


def extract_landmarks(landmarks):
    return np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks])


def interpolate(landmarks):
    for i, data in enumerate(landmarks):
        if np.sum(data) == 0:
            # Find the previous and next non-zero rows
            prev_idx = i - 1 if i > 0 else None
            while prev_idx is not None and np.sum(landmarks[prev_idx]) == 0:
                prev_idx = prev_idx - 1 if prev_idx > 0 else None

            next_idx = i + 1 if i < len(landmarks) - 1 else None
            while next_idx is not None and np.sum(landmarks[next_idx]) == 0:
                next_idx = next_idx + 1 if next_idx < len(landmarks) - 1 else None

            # Ensure there are valid indices
            if prev_idx is not None and next_idx is not None:
                # Interpolation between previous and next non-zero rows
                interpolator = interp1d([prev_idx, next_idx], np.array([landmarks[prev_idx], landmarks[next_idx]]), axis=0, fill_value="extrapolate")
                landmarks[i] = interpolator(i)
            elif prev_idx is not None:
                landmarks[i] = landmarks[prev_idx]
            elif next_idx is not None:
                landmarks[i] = landmarks[next_idx]
    return landmarks


def savgol(landmarks, window_length=15, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth landmark data.

    :param landmarks: numpy array of shape (n_frames, n_landmarks, 3)
    :param window_length: Length of the filter window (must be odd and greater than polyorder)
    :param polyorder: Order of the polynomial used to fit the samples
    :return: Smoothed landmark data
    """
    n_frames, n_landmarks, n_dims = landmarks.shape
    smoothed_landmarks = np.zeros_like(landmarks)

    for i in range(n_landmarks):
        for j in range(n_dims):
            smoothed_landmarks[:, i, j] = savgol_filter(landmarks[:, i, j], window_length, polyorder)

    return smoothed_landmarks


def get_face_options(model_path):
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    # Create a face landmarker instance with the video mode:
    face_options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.VIDEO)

    return face_options


def get_pose_options(model_path):
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    pose_options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.VIDEO)

    return pose_options


def get_hand_options(model_path):
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    hand_options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_hands=1, running_mode=VisionRunningMode.VIDEO)

    return hand_options


def video_frames(video_path):
    """
    Yield (frame, timestamp in ms) for every frame of a video file.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        yield frame, int(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()


def get_landmarks(frames, pose_options, hand_options, face_options):
    """
    Detect the pose, hands and face landmarks of a sequence of (BGR frame, timestamp in ms).
    Hands and face are detected in crops around the pose landmarks.
    """
    with (
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        vision.HandLandmarker.create_from_options(hand_options) as right_hand_landmarker,
        vision.HandLandmarker.create_from_options(hand_options) as left_hand_landmarker,
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
    ):
        # Arrays to store landmarks for all frames
        pose_data = []
        right_hand_data = []
        left_hand_data = []
        face_data = []

        for frame, timestamp in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose landmarks
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp)

            if pose_result.pose_landmarks:
                pose_landmarks = pose_result.pose_landmarks[0]  # Assuming we're dealing with one person

                # Append pose landmarks
                pose_data.append(extract_landmarks(pose_landmarks))

                # Right hand detection (using wrist and pinky as reference)
                right_wrist = pose_landmarks[16]  # Right wrist
                right_pinky = pose_landmarks[22]  # Right pinky
                right_hand_region = [right_wrist, right_pinky]
                right_hand_img, (rx, ry) = center_region(rgb_frame, right_hand_region, padding=100)
                right_hand_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(right_hand_img))
                right_hand_result = right_hand_landmarker.detect_for_video(right_hand_mp, timestamp)

                # Adjust coordinates and draw right hand landmarks
                if right_hand_result.hand_landmarks:
                    right_hand_hand_landmarks = right_hand_result.hand_landmarks[0]
                    right_hand_adjusted_landmarks = []
                    for landmark in right_hand_hand_landmarks:
                        adjusted_x = landmark.x * right_hand_img.shape[1] / rgb_frame.shape[1] + rx / rgb_frame.shape[1]
                        adjusted_y = landmark.y * right_hand_img.shape[0] / rgb_frame.shape[0] + ry / rgb_frame.shape[0]
                        right_hand_adjusted_landmarks.append(type(landmark)(x=adjusted_x, y=adjusted_y, z=landmark.z))
                    right_hand_adjusted_landmarks = extract_landmarks(right_hand_adjusted_landmarks)
                else:
                    right_hand_adjusted_landmarks = np.zeros((21, 3))  # Assuming 21 keypoints for hand
                right_hand_data.append(right_hand_adjusted_landmarks)

                # Left hand detection (using wrist and pinky as reference)
                left_wrist = pose_landmarks[15]  # Left wrist
                left_pinky = pose_landmarks[21]  # Left pinky
                left_hand_region = [left_wrist, left_pinky]
                left_hand_img, (lx, ly) = center_region(rgb_frame, left_hand_region, padding=100)
                left_hand_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(left_hand_img))
                left_hand_result = left_hand_landmarker.detect_for_video(left_hand_mp, timestamp)

                # Adjust coordinates and draw left hand landmarks
                if left_hand_result.hand_landmarks:
                    left_hand_landmarks = left_hand_result.hand_landmarks[0]
                    left_hand_adjusted_landmarks = []
                    for landmark in left_hand_landmarks:
                        adjusted_x = landmark.x * left_hand_img.shape[1] / rgb_frame.shape[1] + lx / rgb_frame.shape[1]
                        adjusted_y = landmark.y * left_hand_img.shape[0] / rgb_frame.shape[0] + ly / rgb_frame.shape[0]
                        left_hand_adjusted_landmarks.append(type(landmark)(x=adjusted_x, y=adjusted_y, z=landmark.z))
                    left_hand_adjusted_landmarks = extract_landmarks(left_hand_adjusted_landmarks)
                else:
                    left_hand_adjusted_landmarks = np.zeros((21, 3))  # Assuming 21 keypoints for hand
                left_hand_data.append(left_hand_adjusted_landmarks)

                # Face detection (using nose and eyes as reference)
                nose = pose_landmarks[0]  # Nose
                left_eye = pose_landmarks[2]  # Left eye
                right_eye = pose_landmarks[5]  # Right eye
                face_region = [nose, left_eye, right_eye]
                face_img, (fx, fy) = center_region(rgb_frame, face_region, padding=100)
                face_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(face_img))
                face_result = face_landmarker.detect_for_video(face_mp, timestamp)

                # Adjust coordinates and draw face landmarks
                if face_result.face_landmarks:
                    face_landmarks = face_result.face_landmarks[0]
                    face_adjusted_landmarks = []
                    for landmark in face_landmarks:
                        adjusted_x = landmark.x * face_img.shape[1] / rgb_frame.shape[1] + fx / rgb_frame.shape[1]
                        adjusted_y = landmark.y * face_img.shape[0] / rgb_frame.shape[0] + fy / rgb_frame.shape[0]
                        face_adjusted_landmarks.append(type(landmark)(x=adjusted_x, y=adjusted_y, z=landmark.z))
                    face_adjusted_landmarks = extract_landmarks(face_adjusted_landmarks)
                else:
                    face_adjusted_landmarks = np.zeros((478, 3))  # Assuming 478 keypoints for facce
                face_data.append(face_adjusted_landmarks)
            else:
                pose_data.append(np.zeros((33, 3)))  # Assuming 33 keypoints for body
                right_hand_data.append(np.zeros((21, 3)))  # Assuming 21 keypoints for hand
                left_hand_data.append(np.zeros((21, 3)))  # Assuming 21 keypoints for hand
                face_data.append(np.zeros((478, 3)))  # Assuming 478 keypoints for facce

    return np.array(pose_data), np.array(right_hand_data), np.array(left_hand_data), np.array(face_data)


def extract_keypoints(frames, video_id, data_dir, options):
    """
    Extract the keypoints of one clip and save them as <data_dir>/poses/{pose,right_hand,left_hand,face}/<video_id>.npy
    """
    landmarks = get_landmarks(frames, *options)

    for part, data in zip(POSE_PARTS, landmarks, strict=True):
        # Interpolate missing detections
        data = interpolate(data)
        # Apply Savitzky-Golay filter to smooth the data (reduce vibrations)
        data = savgol(data)
        np.save(os.path.join(data_dir, 'poses', part, video_id + '.npy'), data)


def load_options(model_dir):
    """
    Create the pose, hand and face landmarker options from the MediaPipe .task models in model_dir.
    """
    return (
        get_pose_options(os.path.join(model_dir, 'pose_landmarker_heavy.task')),
        get_hand_options(os.path.join(model_dir, 'hand_landmarker.task')),
        get_face_options(os.path.join(model_dir, 'face_landmarker.task')),
    )


def make_pose_dirs(data_dir):
    for part in POSE_PARTS:
        os.makedirs(os.path.join(data_dir, 'poses', part), exist_ok=True)


# landmarker options of a worker process, created once per process by _init_worker
_worker_options = None


def _init_worker(model_dir):
    global _worker_options
    _worker_options = load_options(model_dir)


def _extract_clip(job):
    frames_fn, source, clip_id, data_dir = job
    extract_keypoints(frames_fn(source), clip_id, data_dir, _worker_options)
    return clip_id


def extract_clips(clips, data_dir, model_dir, workers=1):
    """
    Extract the keypoints of every (frames_fn, source, clip_id) in clips, where frames_fn(source) yields
    (frame, timestamp in ms). With workers > 1 the clips are split across processes; every clip is still
    processed in order by a single process, so the keypoints are the same as with one worker.
    frames_fn has to be a module-level function so it can be sent to the worker processes.
    """
    make_pose_dirs(data_dir)

    if workers <= 1:
        options = load_options(model_dir)
        for frames_fn, source, clip_id in clips:
            print('Extracting pose for: {}'.format(clip_id))
            extract_keypoints(frames_fn(source), clip_id, data_dir, options)
        return

    jobs = [(frames_fn, source, clip_id, data_dir) for frames_fn, source, clip_id in clips]
    with multiprocessing.Pool(workers, initializer=_init_worker, initargs=(model_dir,)) as pool:
        for done, clip_id in enumerate(pool.imap_unordered(_extract_clip, jobs), 1):
            print('[{}/{}] Extracted pose for: {}'.format(done, len(jobs), clip_id), flush=True)
