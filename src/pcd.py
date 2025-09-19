import open3d as o3d
import numpy as np

# 読み込み
pcd = o3d.io.read_point_cloud("/home/ubuntu/ros2_ws/src/FAST_LIO/PCD/test_xyz.pcd")
points = np.asarray(pcd.points)

# intensity=0 を追加
points_i = np.hstack([points, np.zeros((points.shape[0], 1))])

# 新しい点群に変換
pcd_i = o3d.geometry.PointCloud()
pcd_i.points = o3d.utility.Vector3dVector(points_i[:, :3])

# Open3D は intensity を直接保存できないので、XYZI の PCD を手動で書く
with open("test_xyzi.pcd", "w") as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z intensity\n")
    f.write("SIZE 4 4 4 4\n")
    f.write("TYPE F F F F\n")
    f.write("COUNT 1 1 1 1\n")
    f.write(f"WIDTH {points_i.shape[0]}\n")
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {points_i.shape[0]}\n")
    f.write("DATA ascii\n")
    for p in points_i:
        f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")

print("✅ Saved test_xyzi.pcd with dummy intensity=0")
