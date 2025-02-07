"""Functions to testing the performance of the forward kinematics."""

import torch
import sys

sys.path.append(".")

from examples.world_engine.diff_fk import ForwardKinematics


if __name__ == "__main__":
    urdf_file = "/home/haonan/Project/Piper_ros/src/piper_description/urdf/piper_no_gripper_description.urdf"
    fk = ForwardKinematics(urdf_path=urdf_file, base_link="base_link", ee_link="link5")
