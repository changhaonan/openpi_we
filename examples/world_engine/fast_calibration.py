"""A very fast calibration for top-camera using AprilTag.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import cv2
from pupil_apriltags import Detector
from urchin import URDF

import sys

sys.path.append(".")
import urchin
from examples.world_engine.convert_mcap_data_to_lerobot import read_mcap_file, load_raw_data_from_file


# ===========================================
# Step 1: Load Video and Known Poses
# ===========================================
def load_data(frames, T_base_tag_list):
    # Validate input
    assert len(frames) == len(T_base_tag_list), "Frame count != pose count"
    return frames, T_base_tag_list


# ===========================================
# Step 2: Detect AprilTag Poses from Video
# ===========================================
def detect_T_cam_tag(frames, tag_size, mtx, dist, debug=False):
    detector = Detector("tag36h11")
    all_T_cam_tag = []
    valid_indices = []  # Frames where tag was detected

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            # cv2.imwrite(f"./debug/frame_{idx}_gray.png", gray)
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
                img_pts = det.corners
                ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist)

                if ret:
                    R_cam_tag, _ = cv2.Rodrigues(rvec)
                    T_cam_tag = np.eye(4)
                    T_cam_tag[:3, :3] = R_cam_tag
                    T_cam_tag[:3, 3] = tvec.flatten()
                    all_T_cam_tag.append(T_cam_tag)
                    valid_indices.append(idx)

                    if debug:
                        # Define axis points (origin and the ends of the axes)
                        axis = np.float32(
                            [
                                [0, 0, 0],
                                [tag_size / 2, 0, 0],
                                [0, tag_size / 2, 0],
                                [0, 0, tag_size / 2],
                            ]
                        )
                        # Project 3D points to the image plane
                        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
                        imgpts = np.int32(imgpts).reshape(-1, 2)
                        origin = tuple(imgpts[0])

                        # Draw axes on the frame: X in red, Y in green, Z in blue
                        frame = cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 3)
                        frame = cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 3)
                        frame = cv2.line(frame, origin, tuple(imgpts[3]), (255, 0, 0), 3)

                        cv2.imwrite(f"./debug/frame_{idx}.png", frame)
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
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
        print(f"Final cost: {result.fun}")
        return params_to_transform(result.x)
    else:
        raise RuntimeError("Optimization failed!")


# ===========================================
# Step 4: Main Workflow
# ===========================================
if __name__ == "__main__":
    # Configuration (update these values)
    mcap_file = "/mnt/scratch/datasets/20250207_184745_761519/20250207_184745_761519.mcap"
    urdf_file = "/home/haonan/Project/openpi/examples/world_engine/piper_description/urdf/piper_description.urdf"
    urdf = URDF.load(urdf_file)
    tag_size = 0.1828  # Physical size of the AprilTag in meters
    mtx = np.array([[653.672790527344, 0.0, 645.988403320312], [0.0, 652.861572265625, 362.89111328125], [0.0, 0.0, 1.0]])
    dist = np.array(
        [-0.0537487268447876, 0.0605481714010239, -5.2034647524124e-05, 0.00103097432292998, -0.020617974922061]
    )  # Distortion coefficients

    # Load T_base_tag poses (replace with your known poses)
    T_tag_to_link6 = np.eye(4)  # Measured
    T_tag_to_link6[:3, 3] = np.array([0, 0, (93.4 + 91.4 + 69.2) / 1000.0])
    T_tag_to_link6[:3, :3] = R.from_euler("XYZ", [-90, 180, 0], degrees=True).as_matrix()

    joint_position, ee_poses, action, top_camera_image, left_camera_image, right_camera_image = load_raw_data_from_file(
        mcap_file, fps=50, max_num=5, start_idx=2000, resize=False
    )
    T_tag_to_bases = []
    T_link6_to_bases = []
    for js in joint_position:
        joint_cfg = {
            "joint1": js[0],
            "joint2": js[1],
            "joint3": js[2],
            "joint4": js[3],
            "joint5": js[4],
            "joint6": js[5],
        }
        T_link6_base = urdf.link_fk(cfg=joint_cfg)[urdf.links[6]]
        T_tag_to_base = np.matmul(T_link6_base, T_tag_to_link6)
        T_tag_to_bases.append(T_tag_to_base)
        T_link6_to_bases.append(T_link6_base)
    T_tag_to_bases = np.stack(T_tag_to_bases)  # (N, 4, 4)
    T_link6_to_bases = np.stack(T_link6_to_bases)  # (N, 4, 4)

    # Load video and filter valid poses
    frames, T_tag_to_bases = top_camera_image, T_tag_to_bases
    all_T_cam_tag, valid_indices = detect_T_cam_tag(frames, tag_size, mtx, dist, debug=True)
    filtered_T_tag_to_bases = [T_tag_to_bases[i] for i in valid_indices]

    # Run calibration
    T_cam_base = calibrate_camera_to_base(all_T_cam_tag, filtered_T_tag_to_bases)
    print("Calibrated T_cam_base:\n", T_cam_base)
    np.save("./debug/T_cam_base.npy", T_cam_base)

    # [DEBUG]: Visualize the calibration by projecting the base frame to the camera & robot actions
    # points_3d = np.matmul(np.linalg.inv(T_cam_base)[None, ...], T_link6_to_bases)[:, :3, 3]
    points_3d = np.linalg.inv(T_cam_base)[:3, 3].reshape(-1, 3)
    # Project 3D points to the image plane
    imgpts, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    origin = tuple(imgpts[0])
    # Draw trajectories
    for i in range(1, len(imgpts)):
        cv2.line(frames[0], imgpts[i - 1], tuple(imgpts[i]), (0, 0, 255), 3)
    cv2.imwrite(f"./debug/frame_base.png", frames[0])
