import os
import json
from mcap.reader import make_reader
from datetime import datetime
import bisect
from mcap.decoder import DecoderFactory
import zarr
import numpy as np
import tqdm
import cv2
from multiprocessing import Pool


def get_task_folders(nas_folder, task_type):
    episode_folders = os.listdir(nas_folder)
    task_folders = []
    for episode_folder in episode_folders:
        abs_path = os.path.join(nas_folder, episode_folder)
        metadata_file = os.path.join(abs_path, 'metadata.json')
        if not os.path.exists(metadata_file):
            continue
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if metadata['task_type'] == task_type:
                task_folders.append(episode_folder)
    return task_folders


def read_mcap_file(file_path):
    """
    Read messages from an MCAP file
    """
    messages = []
    with open(file_path, 'rb') as f:
        decoder = DecoderFactory()
        reader = make_reader(f, decoder_factories=decoder)
        for schema, channel, message in reader.iter_messages():
            messages.append({
                'timestamp': message.log_time,  # nanoseconds since Unix epoch
                'data': message.data,
                'channel': channel.topic,
                'schema': schema
            })

    robot_obs = {}
    robot_action = {}
    top_camera = {}
    left_camera = {}
    right_camera = {}
    sync_timestamps = {}

    for message in messages:
        data = json.loads(message['data'].decode())

        if message['channel'] == 'robot_observation':
            robot_obs[message['timestamp']] = data
        elif message['channel'] == 'robot_action':
            if len(data['action']) != 14:
                print(data['action'])
                print(message)
            robot_action[message['timestamp']] = data
        elif message['channel'] == 'top_camera':
            top_camera[message['timestamp']] = data
        elif message['channel'] == 'left_camera':
            left_camera[message['timestamp']] = data
        elif message['channel'] == 'right_camera':
            right_camera[message['timestamp']] = data
        elif message['channel'] == 'sync_timestamps':
            sync_timestamps[message['timestamp']] = data

    return robot_obs, robot_action, top_camera, left_camera, right_camera, sync_timestamps


def resize_image(img_str):
    img_bytes = bytes.fromhex(img_str)
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (320, 240))
    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()


def save_episode_to_zarr(args):
    """Writes data to a separate Zarr sub-group per process to ensure thread safety."""
    mcap_file, zarr_store_path, process_idx = args
    # Create process-specific Zarr store
    store = zarr.DirectoryStore(zarr_store_path)
    root = zarr.group(store=store)
    
    # Create a sub-group for this process
    process_group = root.create_group(f'process_{process_idx}', overwrite=True)
    
    # Define shapes
    joint_positions_shape = (14,)
    ee_pos_shape = (12,)
    action_shape = (14,)

    # Create datasets inside process-specific Zarr group
    joint_position = process_group.create_dataset(
        'joint_position', shape=(0, *joint_positions_shape), dtype=np.float32, maxshape=(None, *joint_positions_shape), chunks=(1, *joint_positions_shape)
    )
    ee_pos = process_group.create_dataset(
        'ee_pos', shape=(0, *ee_pos_shape), dtype=np.float32, maxshape=(None, *ee_pos_shape), chunks=(1, *ee_pos_shape)
    )
    action = process_group.create_dataset(
        'action', shape=(0, *action_shape), dtype=np.float32, maxshape=(None, *action_shape), chunks=(1, *action_shape)
    )
    top_camera_image = process_group.create_dataset(
        'top_camera_image', shape=(0,), dtype=bytes, maxshape=(None,), chunks=True
    )
    left_camera_image = process_group.create_dataset(
        'left_camera_image', shape=(0,), dtype=bytes, maxshape=(None,), chunks=True
    )
    right_camera_image = process_group.create_dataset(
        'right_camera_image', shape=(0,), dtype=bytes, maxshape=(None,), chunks=True
    )

    # Read MCAP file
    robot_obs, robot_action, top_camera, left_camera, right_camera, sync_timestamps = read_mcap_file(mcap_file)
    
    episode_length = 0
    for key in sorted(sync_timestamps.keys()):
        sync_ts = sync_timestamps[key]

        if sync_ts['top_camera_timestamp'] is None or sync_ts['left_camera_timestamp'] is None or sync_ts['right_camera_timestamp'] is None:
            continue

        episode_length += 1
        joint_pos = robot_obs[sync_ts['robot_observation_timestamp']]['joint_positions']
        ee_pos = robot_obs[sync_ts['robot_observation_timestamp']]['ee_pos']
        action_dt = robot_action[sync_ts['robot_action_timestamp']]['action']

        top_cam_img = top_camera[sync_ts['top_camera_timestamp']]['image']
        left_cam_img = left_camera[sync_ts['left_camera_timestamp']]['image']
        right_cam_img = right_camera[sync_ts['right_camera_timestamp']]['image']

        # Append data safely to the process-specific dataset
        joint_position.append(np.array(joint_pos).reshape(1, -1))
        ee_pos.append(np.array(ee_pos).reshape(1, -1))
        action.append(np.array(action_dt).reshape(1, -1))
        top_camera_image.append([resize_image(top_cam_img)])
        left_camera_image.append([resize_image(left_cam_img)])
        right_camera_image.append([resize_image(right_cam_img)])

    return episode_length, f'process_{process_idx}'


def merge_zarr_data(zarr_store_path, process_groups):
    """Merges all process-specific groups into the main dataset."""
    store = zarr.DirectoryStore(zarr_store_path)
    root = zarr.group(store=store)
    
    if 'data' in root:
        del root['data']
    data_group = root.create_group('data')

    if 'meta' in root:
        del root['meta']
    meta_group = root.create_group('meta')

    # Define final datasets
    joint_position = data_group.create_dataset(
        'joint_position', shape=(0, 14), dtype=np.float32, maxshape=(None, 14), chunks=(1, 14)
    )
    ee_pos = data_group.create_dataset(
        'ee_pos', shape=(0, 12), dtype=np.float32, maxshape=(None, 12), chunks=(1, 12)
    )
    action = data_group.create_dataset(
        'action', shape=(0, 14), dtype=np.float32, maxshape=(None, 14), chunks=(1, 14)
    )
    top_camera_image = data_group.create_dataset(
        'top_camera_image', shape=(0,), dtype=bytes, maxshape=(None,), chunks=True
    )
    left_camera_image = data_group.create_dataset(
        'left_camera_image', shape=(0,), dtype=bytes, maxshape=(None,), chunks=True
    )
    right_camera_image = data_group.create_dataset(
        'right_camera_image', shape=(0,), dtype=bytes, maxshape=(None,), chunks=True
    )
    episode_ends = meta_group.create_dataset(
        'episode_ends', shape=(0,), dtype=np.int32, chunks=True
    )

    total_length = 0
    for process_group in process_groups:
        src_group = root[process_group]

        joint_position.append(src_group['joint_position'])
        ee_pos.append(src_group['ee_pos'])
        action.append(src_group['action'])
        top_camera_image.append(src_group['top_camera_image'])
        left_camera_image.append(src_group['left_camera_image'])
        right_camera_image.append(src_group['right_camera_image'])

        total_length += len(src_group['joint_position'])
    
    episode_ends.resize(total_length)
    episode_ends[:] = np.cumsum([len(root[group]['joint_position']) for group in process_groups])

    # [DEBUG]: check the shape for each data
    print(f"joint_position len: {len(joint_position)}.")
    print(f"ee_pos len: {len(ee_pos)}.")
    print(f"action len: {len(action)}.")
    print(f"top_camera_image len: {len(top_camera_image)}.")
    print(f"left_camera_image len: {len(left_camera_image)}.")
    # Clean up process-specific groups
    for group in process_groups:
        del root[group]


def save_all_to_zarr(zarr_path, nas_folder, task_type):
    """Main function that utilizes multiprocessing for thread-safe Zarr writing."""
    task_lists = get_task_folders(nas_folder, task_type)
    print(f'Found {len(task_lists)} tasks')

    # Prepare arguments for parallel processing
    process_args = [
        (os.path.join(nas_folder, task_folder, f'{task_folder}.mcap'), zarr_path, idx)
        for idx, task_folder in enumerate(task_lists)
    ]

    num_processes = min(4, len(task_lists))  # Use up to 4 processes

    with Pool(num_processes) as pool:
        results = list(tqdm.tqdm(pool.imap(save_episode_to_zarr, process_args), total=len(process_args)))

    # Merge results safely
    process_groups = [res[1] for res in results]
    merge_zarr_data(zarr_path, process_groups)

    return [res[0] for res in results]


if __name__ == "__main__":
    nas_folder = '/mnt/scratch/datasets/'
    task_type = 'plate-collection'
    zarr_path = os.path.join('/home/haonan/bc_data/processed_data/', task_type, 'all', '01-28-2025_v2')
    save_all_to_zarr(zarr_path, nas_folder, task_type)
    pass