"""Functions to testing the performance of the forward kinematics."""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from urchin import URDF

import openpi.training.data_loader as _data_loader
import sys

sys.path.append(".")
import numpy as np
import jax
import jax.numpy as jnp
import openpi.training.sharding as sharding
from scipy.spatial.transform import Rotation as R
from examples.world_engine.world_engine_config import _WE_CONFIGS
from examples.world_engine.diff_fk import ForwardKinematics
from examples.world_engine.fast_calibration import load_raw_data_from_file
from examples.world_engine.world_engine_policy import PepperOutputs


def visualize_action(joint_positions, top_camera_image, urdf_file, left_cam2base, right_cam2base, mtx, dist, image_name=""):
    """Visualize the action on the image."""
    # Load URDF
    urdf = URDF.load(urdf_file)
    T_link6_to_bases_left = []
    T_link6_to_bases_right = []
    for js in joint_positions:
        left_joint_cfg = {
            "joint1": js[0],
            "joint2": js[1],
            "joint3": js[2],
            "joint4": js[3],
            "joint5": js[4],
            "joint6": js[5],
        }
        right_joint_cfg = {
            "joint1": js[7],
            "joint2": js[8],
            "joint3": js[9],
            "joint4": js[10],
            "joint5": js[11],
            "joint6": js[12],
        }
        T_link6_base_left = urdf.link_fk(cfg=left_joint_cfg)[urdf.links[6]]
        T_link6_to_bases_left.append(T_link6_base_left)
        T_link6_base_right = urdf.link_fk(cfg=right_joint_cfg)[urdf.links[6]]
        T_link6_to_bases_right.append(T_link6_base_right)
    T_link6_to_bases_left = np.stack(T_link6_to_bases_left)  # (N, 4, 4)
    T_link6_to_bases_right = np.stack(T_link6_to_bases_right)  # (N, 4, 4)
    # Project the link6 to the camera
    T_link6_to_cams_left = np.matmul(np.linalg.inv(left_cam2base), T_link6_to_bases_left)[:, :3, 3]
    T_link6_to_cams_right = np.matmul(np.linalg.inv(right_cam2base), T_link6_to_bases_right)[:, :3, 3]
    # Project the link6 to the image
    left_proj_left = cv2.projectPoints(T_link6_to_cams_left, np.zeros(3), np.zeros(3), mtx, dist)[0].reshape(-1, 2)
    left_proj_right = cv2.projectPoints(T_link6_to_cams_right, np.zeros(3), np.zeros(3), mtx, dist)[0].reshape(-1, 2)
    # Visualize the output by plotting the trajectories on  image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(top_camera_image)
    ax.plot(left_proj_left[:, 0], left_proj_left[:, 1], "rx", label="Left trajectory", markersize=8)
    ax.plot(left_proj_right[:, 0], left_proj_right[:, 1], "bx", label="Right trajectory", markersize=8)
    ax.legend()
    plt.savefig(f"./{image_name}.png") if image_name else plt.savefig("./output.png")
    fig.clear()


def test_fk_from_mcap(urdf_file, mcap_file, left_cam2base, right_cam2base, mtx, dist):
    """Test the forward kinematics using the URDF file."""
    joint_position, ee_poses, action, top_camera_image, left_camera_image, right_camera_image = load_raw_data_from_file(
        mcap_file, fps=50, max_num=None, start_idx=0, resize=False
    )
    visualize_action(joint_position, top_camera_image[0], urdf_file, left_cam2base, right_cam2base, mtx, dist)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    urdf_file = f"{root_dir}/examples/world_engine/piper_description/urdf/piper_description.urdf"
    we_name = "we-003"
    # camera information
    we_cam_dict = {
        "we-003": {
            "left_cam2base": np.load(f"{root_dir}/examples/world_engine/calibration/T_cam_base_left.npy"),
            "right_cam2base": np.load(f"{root_dir}/examples/world_engine/calibration/T_cam_base_right.npy"),
            "mtx": np.array([[653.672790527344, 0.0, 645.988403320312], [0.0, 652.861572265625, 362.89111328125], [0.0, 0.0, 1.0]]),
            "dist": np.array(
                [-0.0537487268447876, 0.0605481714010239, -5.2034647524124e-05, 0.00103097432292998, -0.020617974922061]
            ),  # Distortion coefficients
        }
    }

    ##### Test fk using raw mcap flie ####
    # # mcap_file = f"{root_dir}/test_data/20250205_194514_694214.mcap"
    # mcap_file = "/mnt/scratch/datasets/20250206_164446_007203/20250206_164446_007203.mcap"
    # test_fk_from_mcap(
    #     urdf_file,
    #     mcap_file,
    #     we_cam_dict[we_name]["left_cam2base"],
    #     we_cam_dict[we_name]["right_cam2base"],
    #     we_cam_dict[we_name]["mtx"],
    #     we_cam_dict[we_name]["dist"],
    # )
    ##### Test fk using data-loader ####
    config = _WE_CONFIGS[0]

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    # # Extract data
    batch_idx = 11
    base_0_rgb = batch[0].images["base_0_rgb"][batch_idx]
    state = batch[0].state[:, :14][batch_idx]
    action = batch[1][:, :, :14][batch_idx]
    base_0_rgb = jax.device_get(base_0_rgb)
    base_0_rgb = ((base_0_rgb + 1.0) / 2.0) * 255.0
    base_0_rgb = base_0_rgb.astype(np.uint8)
    # base_0_rgb = cv2.cvtColor(base_0_rgb, cv2.COLOR_RGB2BGR)
    H, W = base_0_rgb.shape[1:3]
    # [DEBUG]
    cv2.imwrite("./base_0_rgb.png", base_0_rgb)
    left_cam2base = we_cam_dict[we_name]["left_cam2base"]
    right_cam2base = we_cam_dict[we_name]["right_cam2base"]
    mtx = we_cam_dict[we_name]["mtx"]
    dist = we_cam_dict[we_name]["dist"]
    # Recover the base_0_rgb to the original size
    base_0_rgb = cv2.resize(base_0_rgb[28:-28, :, :], (1280, 720))
    # Visualize the state
    state_pepper = PepperOutputs()({"actions": state[None, ...]})
    visualize_action(state_pepper["actions"], base_0_rgb, urdf_file, left_cam2base, right_cam2base, mtx, dist, image_name="base_0_rgb_state")
    # Visualize the action
    action_pepper = PepperOutputs()({"actions": action})
    visualize_action(action_pepper["actions"], base_0_rgb, urdf_file, left_cam2base, right_cam2base, mtx, dist, image_name="base_0_rgb_action")

    ##### Test fk using data-set ####
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = _data_loader.create_dataset(data_config, config.model)
    data = dataset[0]
    # Extract the data
    base_0_rgb = data.images["base_0_rgb"]
    state = data.state[:, :14]
    action = data.actions[:, :, :14]
    # Recover the base_0_rgb to the original size
    base_0_rgb = cv2.resize(base_0_rgb[28:-28, :, :], (1280, 720))
    # Visualize the state
    visualize_action(state[None, ...], base_0_rgb, urdf_file, left_cam2base, right_cam2base, mtx, dist, image_name="base_0_rgb_state")
    # Visualize the action
    visualize_action(action, base_0_rgb, urdf_file, left_cam2base, right_cam2base, mtx, dist, image_name="base_0_rgb_action")
