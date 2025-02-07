"""Functions to testing the performance of the forward kinematics."""

import os
import torch
import sys

import openpi.training.data_loader as _data_loader

sys.path.append(".")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import openpi.training.sharding as sharding
from examples.world_engine.world_engine_config import _WE_CONFIGS
from examples.world_engine.diff_fk import ForwardKinematics


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_file = f"{root_dir}/piper_description/urdf/piper_description_v00.urdf"
    fk_left = ForwardKinematics(urdf_path=urdf_file, base_link="base_link", ee_link="link6")
    fk_right = ForwardKinematics(urdf_path=urdf_file, base_link="base_link", ee_link="link6")

    # Test the forward kinematics
    left_joint_positions = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)
    right_joint_positions = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)
    left_ee = fk_left.forward(x=left_joint_positions)
    right_ee = fk_right.forward(x=right_joint_positions)

    extrinsic_left = jnp.eye(4, dtype=jnp.float32)
    extrinsic_right = jnp.eye(4, dtype=jnp.float32)
    intrinsic_left = jnp.array([[455, 0, 320], [0, 455, 240], [0, 0, 1]])  # (3, 3)
    intrinsic_right = jnp.array([[455, 0, 320], [0, 455, 240], [0, 0, 1]])  # (3, 3)

    # Create data loader
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
    # Extract data
    base_0_rgb = batch[0].images["base_0_rgb"]
    state = batch[0].state[:, :14]
    action = batch[1][:, :, :14]
    B = action.shape[0]
    # Forward kinematics
    left_joint_positions = action[:, :, :6].reshape(-1, 6)
    right_joint_positions = action[:, :, 7:13].reshape(-1, 6)
    extrinsic_left = extrinsic_left.reshape(1, 4, 4).repeat(left_joint_positions.shape[0], axis=0)
    intrinsic_left = intrinsic_left.reshape(1, 3, 3).repeat(left_joint_positions.shape[0], axis=0)
    extrinsic_right = extrinsic_right.reshape(1, 4, 4).repeat(right_joint_positions.shape[0], axis=0)
    intrinsic_right = intrinsic_right.reshape(1, 3, 3).repeat(right_joint_positions.shape[0], axis=0)

    left_ee, left_proj = fk_left.forward(x=left_joint_positions, cam_ext=extrinsic_left, cam_int=intrinsic_left)
    right_ee, right_proj = fk_right.forward(x=right_joint_positions, cam_ext=extrinsic_right, cam_int=intrinsic_right)

    # Reshape the output
    base_0_rgb = jax.device_get(base_0_rgb)
    H, W = base_0_rgb.shape[:2]
    left_proj = jax.device_get(left_proj.reshape(B, -1, 2))
    left_proj[:, :, 0] = left_proj[:, :, 0] / W
    left_proj[:, :, 1] = left_proj[:, :, 1] / H
    right_proj = jax.device_get(right_proj.reshape(B, -1, 2))
    right_proj[:, :, 0] = right_proj[:, :, 0] / W
    right_proj[:, :, 1] = right_proj[:, :, 1] / H
    # Visualize the output by plotting the trajectories on the image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(base_0_rgb)
    ax[1].imshow(base_0_rgb)
    for i in range(B):
        ax[0].plot(left_proj[i, :, 0], left_proj[i, :, 1], "r-")
        ax[1].plot(right_proj[i, :, 0], right_proj[i, :, 1], "r-")
    plt.savefig("./output.png")
    fig.clear()
