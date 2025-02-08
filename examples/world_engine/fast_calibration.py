"""A very fast calibration for top-camera using AprilTag.
"""

import os
import open3d as o3d
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import cv2
from pupil_apriltags import Detector
from urchin import URDF
import bisect
import json
from mcap.decoder import DecoderFactory
from mcap.reader import make_reader

import sys

sys.path.append(".")
import urchin

# from examples.world_engine.convert_mcap_data_to_lerobot import read_mcap_file, load_raw_data_from_file


# ===========================================
# Step 0: Load URDF and MCAP Data
def read_mcap_file(file_path):
    """
    Read messages from an MCAP file
    """
    messages = []
    with open(file_path, "rb") as f:
        decoder = DecoderFactory()
        reader = make_reader(f, decoder_factories=decoder)
        for schema, channel, message in reader.iter_messages():
            messages.append(
                {"timestamp": message.log_time, "data": message.data, "channel": channel.topic, "schema": schema}  # nanoseconds since Unix epoch
            )

    robot_obs = {}
    robot_action = {}
    top_camera = {}
    left_camera = {}
    right_camera = {}
    sync_timestamps = {}

    for message in messages:
        data = json.loads(message["data"].decode())

        if message["channel"] == "robot_observation":
            robot_obs[message["timestamp"]] = data
        elif message["channel"] == "robot_action":
            if len(data["action"]) != 14:
                print(data["action"])
                print(message)
            robot_action[message["timestamp"]] = data
        elif message["channel"] == "top_camera":
            top_camera[message["timestamp"]] = data
        elif message["channel"] == "left_camera":
            left_camera[message["timestamp"]] = data
        elif message["channel"] == "right_camera":
            right_camera[message["timestamp"]] = data
        elif message["channel"] == "sync_timestamps":
            sync_timestamps[message["timestamp"]] = data

    return robot_obs, robot_action, top_camera, left_camera, right_camera, sync_timestamps


def decode_image(img_str, resize=True):
    img_bytes = bytes.fromhex(img_str)
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if resize:
        img = cv2.resize(img, (320, 240))
    # Convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def downsample_synced_timestamps(
    sync_timestamps,
    sample_hz=50,
):
    # Downsample to sample_hz
    sample_interval = int(1e9 / sample_hz)  # Convert Hz to nanoseconds interval
    sorted_sync_timestamps = sorted(sync_timestamps.keys())
    start_ts = sorted_sync_timestamps[0]
    end_ts = sorted_sync_timestamps[-1]
    total_duration = end_ts - start_ts
    num_slots = int((total_duration / 1e9) * sample_hz)

    target_timestamps = [start_ts + sample_interval * i for i in range(num_slots)]

    downsampled = {}
    for target_ts in target_timestamps:
        idx = bisect.bisect_left(sorted_sync_timestamps, target_ts)
        if idx == 0:
            closest_ts = sorted_sync_timestamps[0]
        elif idx == len(sorted_sync_timestamps):
            closest_ts = sorted_sync_timestamps[-1]
        else:
            closest_ts = sorted_sync_timestamps[idx - 1]
        downsampled[closest_ts] = sync_timestamps[closest_ts]
    return downsampled


def load_raw_data_from_file(mcap_file: str, fps=50, start_idx=0, max_num=None, resize=True):
    """Load the raw data from cap with a given fps."""
    joint_position = []
    ee_poses = []
    action = []
    top_camera_image = []
    left_camera_image = []
    right_camera_image = []
    # Read MCAP file
    robot_obs, robot_action, top_camera, left_camera, right_camera, sync_timestamps = read_mcap_file(mcap_file)
    # Downsample fps
    ds_sync_timestamps = downsample_synced_timestamps(sync_timestamps=sync_timestamps, sample_hz=fps)
    episode_length = 0
    for idx, key in enumerate(sorted(ds_sync_timestamps.keys())):
        print(f"Processing frame {idx} / {len(ds_sync_timestamps)}")
        if idx < start_idx:
            continue
        if max_num is not None and idx >= (start_idx + max_num):
            break
        sync_ts = ds_sync_timestamps[key]

        if sync_ts["top_camera_timestamp"] is None or sync_ts["left_camera_timestamp"] is None or sync_ts["right_camera_timestamp"] is None:
            continue

        episode_length += 1
        joint_pos = robot_obs[sync_ts["robot_observation_timestamp"]]["joint_positions"]
        print(joint_pos[7:13])
        ee_pos = robot_obs[sync_ts["robot_observation_timestamp"]]["ee_pos"]
        action_dt = robot_action[sync_ts["robot_action_timestamp"]]["action"]

        top_cam_img = top_camera[sync_ts["top_camera_timestamp"]]["image"]
        left_cam_img = left_camera[sync_ts["left_camera_timestamp"]]["image"]
        right_cam_img = right_camera[sync_ts["right_camera_timestamp"]]["image"]

        # Append data safely to the process-specific dataset
        joint_position.append(np.array(joint_pos).reshape(1, -1))
        ee_poses.append(np.array(ee_pos).reshape(1, -1))
        action.append(np.array(action_dt).reshape(1, -1))
        top_camera_image.append(decode_image(top_cam_img, resize=resize))
        left_camera_image.append(decode_image(left_cam_img, resize=resize))
        right_camera_image.append(decode_image(right_cam_img, resize=resize))

    # [DEBUG]
    joint_position = np.concatenate(joint_position, axis=0)
    ee_poses = np.concatenate(ee_poses, axis=0)
    action = np.concatenate(action, axis=0)
    top_camera_image = np.stack(top_camera_image, axis=0)
    left_camera_image = np.stack(left_camera_image, axis=0)
    right_camera_image = np.stack(right_camera_image, axis=0)
    return joint_position, ee_poses, action, top_camera_image, left_camera_image, right_camera_image


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
                    # Flip the z-axis as camera z-axis is pointing outside
                    # T_cam_tag[2, 3] = -T_cam_tag[2, 3]
                    # Flip the x-axis as camera x-axis is pointing left
                    # T_cam_tag[0, 3] = -T_cam_tag[0, 3]
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


def transform_to_params(T):
    t = T[:3, 3]
    axis_angle = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([t, axis_angle])


def cost_function(params, all_T_tag2cam, all_T_tag2base):
    T_cam2base = params_to_transform(params)
    total_error = 0.0
    for T_tag2cam, T_tag2base in zip(all_T_tag2cam, all_T_tag2base):
        T_tag2base_pred = T_cam2base @ T_tag2cam
        t_error = T_tag2base[:3, 3] - T_tag2base_pred[:3, 3]
        R_true = R.from_matrix(T_tag2base[:3, :3])
        R_pred = R.from_matrix(T_tag2base_pred[:3, :3])
        angle_error = (R_true.inv() * R_pred).magnitude()
        total_error += np.sum(t_error**2) + 10.0 * (angle_error**2)
    return total_error


def calibrate_camera_to_base(all_T_cam_tag, all_T_base_tag, initial_params=None):
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
    hand = "right"  # "left" or "right"
    # Configuration (update these values)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    mcap_file = f"{root_dir}/../../test_data/20250207_184745_761519.mcap"
    urdf_file = f"{root_dir}/piper_description/urdf/piper_description.urdf"
    urdf = URDF.load(urdf_file)
    tag_size = 0.1828  # Physical size of the AprilTag in meters
    mtx = np.array([[653.672790527344, 0.0, 645.988403320312], [0.0, 652.861572265625, 362.89111328125], [0.0, 0.0, 1.0]])
    dist = np.array(
        [-0.0537487268447876, 0.0605481714010239, -5.2034647524124e-05, 0.00103097432292998, -0.020617974922061]
    )  # Distortion coefficients

    # Load T_base_tag poses (replace with your known poses)
    T_tag_to_link6 = np.eye(4)  # Measured
    T_tag_to_link6[:3, 3] = np.array([0, 0, (93.4 + 91.4 + 69.2) / 1000.0])
    T_tag_to_link6[:3, :3] = R.from_euler("XYZ", [90, 180, 0], degrees=True).as_matrix()

    joint_position, ee_poses, action, top_camera_image, left_camera_image, right_camera_image = load_raw_data_from_file(
        mcap_file, fps=50, max_num=500, start_idx=2500, resize=False
    )
    offset = 0 if hand == "left" else 7
    # [DEBUG]: plot joint positions
    import matplotlib.pyplot as plt

    for i in range(6):
        plt.plot(joint_position[:, i + offset], label=f"joint{i+1+offset}")
    plt.show()

    T_tag_to_bases = []
    T_link6_to_bases = []

    for js in joint_position:
        joint_cfg = {
            "joint1": js[0 + offset],
            "joint2": js[1 + offset],
            "joint3": js[2 + offset],
            "joint4": js[3 + offset],
            "joint5": js[4 + offset],
            "joint6": js[5 + offset],
        }
        T_link6_base = urdf.link_fk(cfg=joint_cfg)[urdf.links[6]]
        T_link6_to_bases.append(T_link6_base)
        T_tag_to_base = np.matmul(T_link6_base, T_tag_to_link6)
        T_tag_to_bases.append(T_tag_to_base)
    T_tag_to_bases = np.stack(T_tag_to_bases)  # (N, 4, 4)
    T_link6_to_bases = np.stack(T_link6_to_bases)  # (N, 4, 4)
    # # [DEBUG]: Visualize link6 to bases
    # vis = []
    # for T_link6_base in T_link6_to_bases:
    #     link6_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    #     link6_base.transform(T_link6_base)
    #     vis.append(link6_base)
    # vis.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    # o3d.visualization.draw_geometries(vis)

    # Load video and filter valid poses
    frames, T_tag_to_bases = top_camera_image, T_tag_to_bases
    all_T_tag2cam, valid_indices = detect_T_cam_tag(frames, tag_size, mtx, dist, debug=True)
    filtered_T_tag_to_bases = [T_tag_to_bases[i] for i in valid_indices]

    init_cam2base = np.eye(4)  # camera to base
    init_cam2base[:3, 3] = np.array([0.530225, 0.474345, 1.0287])
    init_cam2base[:3, :3] = R.from_euler("XYZ", [155, 0, 180], degrees=True).as_matrix()
    # [DEBUG]: Visualize the detected poses
    vis = []
    for i, T_tag2cam in enumerate(all_T_tag2cam):
        tag = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        tag.transform(init_cam2base @ T_tag2cam)
        vis.append(tag)
    # Add cam pose
    cam_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    cam_pose.transform(init_cam2base)
    vis.append(cam_pose)
    for T_tag_to_base in T_tag_to_bases:
        tag_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        tag_pose.transform(T_tag_to_base)
        vis.append(tag_pose)
    vis.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    vis.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03))
    o3d.visualization.draw_geometries(vis)
    # Run calibration
    T_cam2base = calibrate_camera_to_base(all_T_tag2cam, filtered_T_tag_to_bases, initial_params=transform_to_params(init_cam2base))
    print("Calibrated T_cam_base:\n", T_cam2base)
    np.save("./debug/T_cam_base.npy", T_cam2base)

    # [DEBUG]: Visualize the detected poses
    vis = []
    for i, T_tag2cam in enumerate(all_T_tag2cam):
        tag = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        tag.transform(T_cam2base @ T_tag2cam)
        vis.append(tag)
    # Add cam pose
    cam_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    cam_pose.transform(T_cam2base)
    vis.append(cam_pose)
    for T_tag_to_base in T_tag_to_bases:
        tag_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        tag_pose.transform(T_tag_to_base)
        vis.append(tag_pose)
    vis.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    vis.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03))
    o3d.visualization.draw_geometries(vis)

    # [DEBUG]: Visualize the calibration by projecting the base frame to the camera & robot actions
    # points_3d = np.matmul(np.linalg.inv(T_cam_base)[None, ...], T_link6_to_bases)[:, :3, 3]
    points_3d = np.linalg.inv(T_cam2base)[:3, 3].reshape(-1, 3)
    # Project 3D points to the image plane
    imgpts, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    origin = tuple(imgpts[0])
    # Draw trajectories
    for i in range(1, len(imgpts)):
        cv2.line(frames[0], imgpts[i - 1], tuple(imgpts[i]), (0, 0, 255), 3)
    cv2.imwrite(f"./debug/frame_base.png", frames[0])
