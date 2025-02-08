"""
Script to convert World Engine's data (zarr) to the LeRobot dataset v2.0 format.

Example usage: uv run examples/world_engine/convert_mcap_data_to_lerobot.py --raw-dir /mnt/scratch/datasets/ --repo-id changhaonan/world_engine_plate
"""

import csv
import os
import dataclasses
from pathlib import Path
import shutil
from typing import Literal
from mcap.reader import make_reader
from datetime import datetime
import bisect
from mcap.decoder import DecoderFactory
import json
import cv2

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps=50,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (240, 320, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


############################## Load raw data from mcap ##############################
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


def resize_image(img_str, resize=True):
    img_bytes = bytes.fromhex(img_str)
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if resize:
        img = cv2.resize(img, (320, 240))
    # Convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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
        ee_pos = robot_obs[sync_ts["robot_observation_timestamp"]]["ee_pos"]
        action_dt = robot_action[sync_ts["robot_action_timestamp"]]["action"]

        top_cam_img = top_camera[sync_ts["top_camera_timestamp"]]["image"]
        left_cam_img = left_camera[sync_ts["left_camera_timestamp"]]["image"]
        right_cam_img = right_camera[sync_ts["right_camera_timestamp"]]["image"]

        # Append data safely to the process-specific dataset
        joint_position.append(np.array(joint_pos).reshape(1, -1))
        ee_poses.append(np.array(ee_pos).reshape(1, -1))
        action.append(np.array(action_dt).reshape(1, -1))
        top_camera_image.append(resize_image(top_cam_img, resize=resize))
        left_camera_image.append(resize_image(left_cam_img, resize=resize))
        right_camera_image.append(resize_image(right_cam_img, resize=resize))

    # [DEBUG]
    joint_position = np.concat(joint_position, axis=0)
    ee_poses = np.concat(ee_poses, axis=0)
    action = np.concat(action, axis=0)
    top_camera_image = np.stack(top_camera_image, axis=0)
    left_camera_image = np.stack(left_camera_image, axis=0)
    right_camera_image = np.stack(right_camera_image, axis=0)
    return joint_position, ee_poses, action, top_camera_image, left_camera_image, right_camera_image


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: list[Path],
    task: str,
    filter_file: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    # Parse all .cap files
    mcap_files = []
    # Read csv file
    run_ids = []
    with open(filter_file, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Each row is a list of strings
            run_ids.append(row[0])
    if run_ids:
        dirs = run_ids
    else:
        dirs = os.listdir(raw_dir)
    for d in dirs:
        if os.path.exists(os.path.join(raw_dir, d, f"{d}.mcap")):
            # read meta
            meta_file = os.path.join(raw_dir, d, "metadata.json")
            with open(meta_file) as f:
                config = json.load(f)
                if config["task_type"] == "plate-collection":
                    mcap_files.append(os.path.join(raw_dir, d, f"{d}.mcap"))
    # # [Sanity check]: Use a small chunk for data checking.
    # mcap_files = mcap_files[:2]

    if episodes is None:
        episodes = range(len(mcap_files))

    for ep_idx in tqdm.tqdm(episodes):
        mcap_file = mcap_files[ep_idx]
        state, ee_poses, action, top_camera_image, left_camera_image, right_camera_image = load_raw_data_from_file(mcap_file)
        num_frames = len(state)
        print(f"num frames: {num_frames}")

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            frame["observation.images.cam_high"] = top_camera_image[i]
            frame["observation.images.cam_left_wrist"] = left_camera_image[i]
            frame["observation.images.cam_right_wrist"] = right_camera_image[i]
            dataset.add_frame(frame)

        dataset.save_episode(task=task, encode_videos=True)

    return dataset


def process_data(
    raw_dir: Path,
    repo_id: str,
    task: str,
    filter_file: str = "",
    fps: int = 50,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="pepper",
        mode=mode,
        dataset_config=dataset_config,
        fps=fps,
    )

    dataset = populate_dataset(
        dataset,
        raw_dir,
        task=task,
        filter_file=filter_file,
        episodes=episodes,
    )
    dataset.consolidate()

    # if push_to_hub:
    #     dataset.push_to_hub()


if __name__ == "__main__":
    # tyro.cli(port_aloha)
    mode = "video"
    task = "plate_collection"
    filter_file = "/home/dihuang/robotics/episode_folders.csv"  # Indicating the selection file
    fps = 50
    process_data(
        raw_dir=Path("/mnt/scratch/datasets/"), repo_id="changhaonan/world_engine_plate", mode=mode, task=task, filter_file=filter_file, fps=fps
    )
