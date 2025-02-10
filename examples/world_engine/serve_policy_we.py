import dataclasses
from openpi.serving import websocket_policy_server
import openpi.training.data_loader as _data_loader
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config
import sys
import os
import socket
from pathlib import Path

sys.path.append(".")
from examples.world_engine.world_engine_config import _WE_CONFIGS


def main(config: _config.TrainConfig, checkpoint_dir: Path) -> None:
    data_config = config.data.create(config.assets_dirs, config.model)
    # Create policy
    policy = _policy_config.create_trained_policy(config, checkpoint_dir=checkpoint_dir, repack_transforms=data_config.repack_transforms)
    policy_metadata = policy.metadata

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port="8000",
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    config = _WE_CONFIGS[0]
    main(config=config, checkpoint_dir=Path("checkpoints/pi0_plate_collect/plate_collect/29999/"))
    # The data format:
    # {
    #     "observation.images.cam_high": [3, 224, 224],
    #     "observation.images.cam_left_wrist": [3, 224, 224],
    #     "observation.images.cam_right_wrist": [3, 224, 224],
    #     "observation.state": [14],
    #     "action": [50, 14],
    # }
