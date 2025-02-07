"""Differentiable Forward kinematics."""

import torch
import torch.nn as nn
from urchin import URDF


def skew_symmetric_matrix(axis):
    """Generate skew-symmetric matrix for rotation axis"""
    K = torch.zeros(3, 3, device=axis.device)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]
    return K


class ForwardKinematics(nn.Module):
    def __init__(self, urdf_path, base_link, ee_link):
        super().__init__()
        # Load URDF and build kinematic chain
        self.urdf = URDF.load(urdf_path)
        self.base_link = base_link
        self.ee_link = ee_link

        # Build parent joint map and find kinematic chain
        self.parent_map = self._build_parent_map()
        self.chain_joints = self._find_chain()

        # Extract joint parameters
        self.transforms = nn.ParameterList()
        self.joint_axes = []
        self.joint_types = []
        self.active_joints = []

        # Preprocess joints in the chain
        for joint in self.chain_joints:
            # Store fixed transform from joint origin
            origin = torch.tensor(joint.origin, dtype=torch.float32)
            self.transforms.append(nn.Parameter(origin, requires_grad=False))

            # Store joint properties
            self.joint_types.append(joint.joint_type)
            if joint.joint_type in ["revolute", "prismatic", "continuous"]:
                axis = torch.tensor(joint.axis, dtype=torch.float32)
                axis /= torch.norm(axis)  # Normalize
                self.joint_axes.append(nn.Parameter(axis, requires_grad=False))
                self.active_joints.append(joint)

        self.num_joints = len(self.active_joints)

    def _build_parent_map(self):
        """Create mapping from child links to their parent joints"""
        return {j.child: j for j in self.urdf.joints}

    def _find_chain(self):
        """Find joint sequence from base to end effector"""
        chain = []
        current = self.urdf.link_map[self.ee_link]

        while current.name != self.base_link:
            if current.name not in self.parent_map:
                raise ValueError(f"No path from {self.base_link} to {self.ee_link}")
            parent_joint = self.parent_map[current.name]
            chain.append(parent_joint)
            current = self.urdf.link_map[parent_joint.parent]

        # Reverse to get base-to-ee order
        return list(reversed(chain))

    def forward(self, joint_positions):
        device = joint_positions.device
        batch_size = joint_positions.shape[0]

        # Initialize transform with identity matrix
        T = torch.eye(4, device=device).repeat(batch_size, 1, 1)

        joint_idx = 0  # Index for active joints

        for i, joint in enumerate(self.chain_joints):
            # Get fixed transform from URDF
            fixed_tf = self.transforms[i].to(device)
            fixed_tf = fixed_tf.unsqueeze(0).repeat(batch_size, 1, 1)

            # Apply joint transform if active
            if joint.joint_type in ["revolute", "continuous"]:
                theta = joint_positions[:, joint_idx]
                axis = self.joint_axes[joint_idx].to(device)

                # Compute rotation using Rodrigues' formula
                K = skew_symmetric_matrix(axis)
                I = torch.eye(3, device=device)
                theta = theta.view(-1, 1, 1)
                R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K

                joint_tf = torch.eye(4, device=device).repeat(batch_size, 1, 1)
                joint_tf[:, :3, :3] = R
                joint_idx += 1

            elif joint.joint_type == "prismatic":
                d = joint_positions[:, joint_idx]
                axis = self.joint_axes[joint_idx].to(device)

                joint_tf = torch.eye(4, device=device).repeat(batch_size, 1, 1)
                joint_tf[:, :3, 3] = d.view(-1, 1) * axis
                joint_idx += 1

            else:
                joint_tf = torch.eye(4, device=device).repeat(batch_size, 1, 1)

            # Compose transformations
            T = T @ fixed_tf @ joint_tf

        return T
