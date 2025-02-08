"""A very fast calibration for top-camera using AprilTag.
"""

import numpy as np
import mediapy as media
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import cv2
from apriltag import apriltag


# ===========================================
# Step 1: Load Video and Known Poses
# ===========================================
def load_data(video_path, T_base_tag_list):
    # Read video frames
    video = media.read_video(video_path)
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in video[0]]  # Convert to BGR

    # Validate input
    assert len(frames) == len(T_base_tag_list), "Frame count != pose count"
    return frames, T_base_tag_list


# ===========================================
# Step 2: Detect AprilTag Poses from Video
# ===========================================
def detect_T_cam_tag(frames, tag_size, mtx, dist):
    detector = apriltag("tag36h11")
    all_T_cam_tag = []
    valid_indices = []  # Frames where tag was detected

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)

        if detections:
            det = detections[0]
            # Define tag corners in 3D (adjust based on your tag's physical size)
            obj_pts = np.array(
                [
                    [-tag_size / 2, -tag_size / 2, 0],
                    [tag_size / 2, -tag_size / 2, 0],
                    [tag_size / 2, tag_size / 2, 0],
                    [-tag_size / 2, tag_size / 2, 0],
                ],
                dtype=np.float32,
            )

            # Solve PnP to get pose
            img_pts = np.array(det["lb-rb-rt-lt"], dtype=np.float32).reshape(4, 2)
            ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist)

            if ret:
                R_cam_tag, _ = cv2.Rodrigues(rvec)
                T_cam_tag = np.eye(4)
                T_cam_tag[:3, :3] = R_cam_tag
                T_cam_tag[:3, 3] = tvec.flatten()
                all_T_cam_tag.append(T_cam_tag)
                valid_indices.append(idx)

    return all_T_cam_tag, valid_indices


# ===========================================
# Step 3: Optimization (Same as Before)
# ===========================================
def params_to_transform(params):
    t = params[:3]
    axis_angle = params[3:]
    R_mat = R.from_rotvec(axis_angle).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def cost_function(params, all_T_cam_tag, all_T_base_tag):
    T_cam_base = params_to_transform(params)
    total_error = 0.0
    for T_cam_tag, T_base_tag in zip(all_T_cam_tag, all_T_base_tag):
        T_pred = T_cam_base @ T_base_tag
        t_error = T_cam_tag[:3, 3] - T_pred[:3, 3]
        R_true = R.from_matrix(T_cam_tag[:3, :3])
        R_pred = R.from_matrix(T_pred[:3, :3])
        angle_error = (R_true.inv() * R_pred).magnitude()
        total_error += np.sum(t_error**2) + 10.0 * (angle_error**2)
    return total_error


def calibrate_camera_to_base(all_T_cam_tag, all_T_base_tag):
    initial_params = np.zeros(6)
    result = minimize(cost_function, initial_params, args=(all_T_cam_tag, all_T_base_tag), method="L-BFGS-B", options={"maxiter": 1000})
    if result.success:
        return params_to_transform(result.x)
    else:
        raise RuntimeError("Optimization failed!")


# ===========================================
# Step 4: Main Workflow
# ===========================================
if __name__ == "__main__":
    # Configuration (update these values)
    video_path = "robot_april_tag.mp4"
    tag_size = 0.1  # Physical size of the AprilTag in meters
    mtx = np.load("calibration_data.npz")["mtx"]  # Camera intrinsics
    dist = np.load("calibration_data.npz")["dist"]  # Distortion coefficients

    # Load T_base_tag poses (replace with your known poses)
    # Example: T_base_tag_list[i] = 4x4 transformation matrix for frame i
    T_base_tag_list = [
        np.eye(4),  # Replace with actual poses
        np.eye(4),
        # ...
    ]

    # Load video and filter valid poses
    frames, T_base_tag_list = load_data(video_path, T_base_tag_list)
    all_T_cam_tag, valid_indices = detect_T_cam_tag(frames, tag_size, mtx, dist)
    filtered_T_base_tag = [T_base_tag_list[i] for i in valid_indices]

    # Run calibration
    T_cam_base = calibrate_camera_to_base(all_T_cam_tag, filtered_T_base_tag)
    print("Calibrated T_cam_base:\n", T_cam_base)
    np.save("T_cam_base.npy", T_cam_base)
