"""Visualize the alignment."""

from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d


if __name__ == "__main__":
    # top camera to left base
    trans = np.array([0.20875, 0.18675, 0.405])
    top2left_base = np.eye(4, dtype=np.float32)
    top2left_base[:3, 3] = trans
    local_rot = np.eye(4, dtype=np.float32)
    local_rot[:3, :3] = R.from_euler("XYZ", [-25, 0, 0], degrees=True).as_matrix()
    top2left_base = top2left_base @ local_rot
    # right base to left base
    trans = np.array([0.4175, 0.0, 0.0])
    right2left_base = np.eye(4, dtype=np.float32)
    right2left_base[:3, 3] = trans
    local_rot = np.eye(4, dtype=np.float32)
    local_rot[:3, :3] = R.from_euler("XYZ", [0, 0, 180], degrees=True).as_matrix()
    right2left_base = right2left_base @ local_rot

    # Visualize everything from the top camera
    vis = []
    vis.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))  # representing top-camera
    # left base
    left_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    left_base.transform(np.linalg.inv(top2left_base))
    vis.append(left_base)
    # # right base
    right_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    right_base.transform(np.linalg.inv(top2left_base) @ right2left_base)
    vis.append(right_base)
    # visualize
    o3d.visualization.draw_geometries(vis)
