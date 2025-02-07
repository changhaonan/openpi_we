"""Differentiable Forward kinematics."""

import jax.numpy as jnp
import torch.nn as nn
from urchin import URDF
from jax import vmap


class ForwardKinematics:
    def __init__(self, urdf_path, base_link, ee_link):
        # Load URDF and build kinematic chain
        self.urdf = URDF.load(urdf_path)
        self.base_link = base_link
        self.ee_link = ee_link

        # Build parent joint map and find kinematic chain
        self.parent_map = self._build_parent_map()
        self.chain_joints = self._find_chain()

        # Extract joint parameters
        self.transforms = []
        self.joint_axes = []
        self.joint_types = []
        self.active_joints = []

        # Preprocess joints in the chain
        for joint in self.chain_joints:
            # Store fixed transform from joint origin
            origin = jnp.array(joint.origin, dtype=jnp.float32)
            self.transforms.append(origin)

            # Store joint properties
            self.joint_types.append(joint.joint_type)
            if joint.joint_type in ["revolute", "prismatic", "continuous"]:
                axis = jnp.array(joint.axis, dtype=jnp.float32)
                axis /= jnp.linalg.norm(axis)  # Normalize
                self.joint_axes.append(axis)
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

    def forward(self, x, cam_ext=None, cam_int=None):
        """Compute forward kinematics for given joint angles, optionally with camera projection.
        Args:
            x (jnp.array): Joint angles or displacements, (batch_size, num_joints).
            cam_ext (jnp.array): Camera extrinsic matrix, (batch_size, 4, 4).
            cam_int (jnp.array): Camera intrinsic matrix, (batch_size, 3, 3).
        """
        batch_size = x.shape[0]

        # Initialize transform with identity matrix
        T = jnp.eye(4).reshape(1, 4, 4).repeat(batch_size, axis=0)

        joint_idx = 0  # Index for active joints

        for i, joint in enumerate(self.chain_joints):
            # Get fixed transform from URDF
            fixed_tf = self.transforms[i].reshape(1, 4, 4).repeat(batch_size, axis=0)

            # Apply joint transform if active
            if joint.joint_type in ["revolute", "continuous"]:
                theta = x[:, joint_idx]
                axis = self.joint_axes[joint_idx]

                # Compute rotation using Rodrigues' formula
                K = skew_symmetric_matrix(axis)
                I = jnp.eye(3)
                theta = theta.reshape(-1, 1, 1)
                R = I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * K @ K

                joint_tf = jnp.eye(4).reshape(1, 4, 4).repeat(batch_size, axis=0)
                joint_tf = joint_tf.at[:, :3, :3].set(R)
                joint_idx += 1

            elif joint.joint_type == "prismatic":
                d = x[:, joint_idx]
                axis = self.joint_axes[joint_idx]

                joint_tf = jnp.eye(4).reshape(1, 4, 4).repeat(batch_size, axis=0)
                joint_tf = joint_tf.at[:, :3, 3].set(d.reshape(-1, 1) * axis)
                joint_idx += 1

            else:
                joint_tf = jnp.eye(4).reshape(1, 4, 4).repeat(batch_size, axis=0)

            # Compose transformations
            T = T @ fixed_tf @ joint_tf

        if cam_ext is not None and cam_int is not None:
            # When cam_ext/cam_int is provided, we want to project the ee position to pixel space
            p = T[:, :3, 3]
            p = jnp.matmul(jnp.linalg.inv(cam_ext), jnp.concatenate([p, jnp.ones((batch_size, 1))], axis=1).reshape(batch_size, 4, 1)).reshape(
                batch_size, 4
            )
            p = p[:, :3] / (p[:, 2:3] + 1e-6)
            p_proj = jnp.matmul(cam_int, p.reshape(batch_size, 3, 1))[:, :2, 0]  # (batch_size, 3)
            return T, p_proj
        return T, None


def skew_symmetric_matrix(axis):
    """Compute the skew-symmetric matrix for a given axis."""
    return jnp.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=jnp.float32)
