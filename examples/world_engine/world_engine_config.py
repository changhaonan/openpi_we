from collections.abc import Sequence
import dataclasses
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override

from openpi.training.config import TrainConfig, DataConfig, DataConfigFactory, ModelTransformFactory
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# Local depend
import examples.world_engine.world_engine_policy as world_engine_policy


@dataclasses.dataclass(frozen=True)
class LeRobotWEDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the WE environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/cam_high": "cam_high",
                        "observation/cam_left_wrist": "cam_left_wrist",
                        "observation/cam_right_wrist": "cam_right_wrist",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[world_engine_policy.PepperInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[world_engine_policy.PepperOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


#################################### Data classes ####################################
_WE_CONFIGS = [
    TrainConfig(
        name="pi0_peper_we",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotWEDataConfig(
            repo_id="changhaonan/world_engine_plate",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        ema_decay=None,
    )
]
