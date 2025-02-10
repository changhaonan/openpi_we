"""Inference test for World Engine."""

import numpy as np
import dataclasses
import functools
import logging
import platform
from typing import Any
from pathlib import Path
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb
import cv2
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.policies import policy_config as _policy_config
import sys
import os

sys.path.append(".")
from examples.world_engine.world_engine_config import _WE_CONFIGS
from examples.world_engine.vis_tools import output_transform, visualize_action


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def main(config: _config.TrainConfig, checkpoint_dir):
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
    left_cam2base = we_cam_dict[we_name]["left_cam2base"]
    right_cam2base = we_cam_dict[we_name]["right_cam2base"]
    mtx = we_cam_dict[we_name]["mtx"]
    dist = we_cam_dict[we_name]["dist"]

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    key = jax.random.key(0)
    # Create a model from the checkpoint.
    # model = config.model.load(_model.restore_params(checkpoint_dir / "params"))
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_dataset(data_config, config.model)
    # Create policy
    policy = _policy_config.create_trained_policy(config, checkpoint_dir=checkpoint_dir, repack_transforms=data_config.repack_transforms)
    # Create example
    sample_idxes = np.random.randint(0, len(dataset), size=10)
    for sample_idx in sample_idxes:
        data = dataset[int(sample_idx)]
        result = policy.infer(data)
        # Extract the data
        actions = result["actions"]
        actions_gt = data["action"].cpu().numpy()
        state = data["observation.state"].cpu().numpy()
        cam_rgb = data["observation.images.cam_high"].cpu().numpy().transpose(1, 2, 0)
        # Resize the image to the original size
        cam_rgb = cv2.resize(cam_rgb, (1280, 720))
        visualize_action(actions, cam_rgb, urdf_file, left_cam2base, right_cam2base, mtx, dist, image_name=f"base_0_rgb_action_{sample_idx}")
        visualize_action(actions_gt, cam_rgb, urdf_file, left_cam2base, right_cam2base, mtx, dist, image_name=f"base_0_rgb_action_gt_{sample_idx}")


if __name__ == "__main__":
    config = _WE_CONFIGS[0]
    main(config=config, checkpoint_dir=Path("checkpoints/pi0_plate_collect/plate_collect/29999/"))
