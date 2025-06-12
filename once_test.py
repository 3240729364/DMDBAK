# # 计算相机到标定板转换矩阵
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# w = 10
# h = 6
#
# # 参数1
# # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵，认为在棋盘格这个平面上Z=0
# objp = np.zeros((w * h, 3), np.float32)  # 构造 0 矩阵，88行3列，用于存放角点的世界坐标
# objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 三维网格坐标划分
# objq = objp * 25  # 将坐标缩放到 25mm 单位
# # 获取图像
#
# # 参数2-------------未解决
# # 输入灰度图并找出角点
# checkerboard_size = (10, 6)
# refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# camera_color_img, camera_depth_img = robot.get_camera_data()
#
# print(type(camera_color_img))
#
# bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
#
# gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
# checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
#                                                         cv2.CALIB_CB_ADAPTIVE_THRESH)
# corners_refined = cv2.cornerSubPix(gray_data, corners, (5,5), (-1,-1), refine_criteria)
# print(type(corners_refined))
#
# # 参数3,4--直接输入
# # 相机内参矩阵
# camera_matrix = np.array([
#     [800, 0, 320],
#     [0, 800, 240],
#     [0, 0, 1]
# ], dtype=np.float32)
#
# # 畸变系数（假设无畸变）
# dist_coeffs = np.zeros(4)
#
#
# # 使用solvePnP求解相机姿态
# success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
#
# # 将旋转向量转换为旋转矩阵
# rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
#
# print("旋转矩阵:\n", rotation_matrix)
# print("平移向量:\n", translation_vector)
# # 重投影3D点以验证结果
# projected_points, _ = cv2.projectPoints(object_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
#
# # 计算重投影误差
# reprojection_error = np.mean(np.linalg.norm(image_points - projected_points.squeeze(), axis=1))
# print("重投影误差:", reprojection_error)
#
#
# # 创建一个空白图像
# image = np.zeros((480, 640, 3), dtype=np.uint8)
#
# # 绘制2D点和重投影点
# for pt, proj_pt in zip(image_points, projected_points):
#     cv2.circle(image, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
#     cv2.circle(image, tuple(proj_pt.squeeze().astype(int)), 5, (0, 0, 255), -1)
#
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("2D Points and Reprojected Points")
# plt.show()
#
#
#
#
# # DEBUG:   机械臂夹持针头找角点（也可以用这个方法先调误差，并且也可以调整位姿）

# import open3d as o3d
#
# # 读取 PCD 文件
# pcd = o3d.io.read_point_cloud("rgbd_point_cloud.pcd")
#
# # 打印点云信息
# print(pcd)  # 输出点云的基本信息（点数、维度等）
#
# # 可视化点云
# o3d.visualization.draw_geometries([pcd])
#
# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import open3d as o3d
#
# # 1. 创建 pipeline
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#
# # 2. 启动流
# pipeline.start(config)
#
# # 3. 创建对齐对象，将深度图对齐到彩色图
# align_to = rs.stream.color
# align = rs.align(align_to)
#
# try:
#     for _ in range(30):  # 等待相机稳定
#         pipeline.wait_for_frames()
#
#     # 获取对齐帧
#     frames = pipeline.wait_for_frames()
#     aligned_frames = align.process(frames)
#
#     # 4. 获取对齐后的彩色图和深度图
#     aligned_depth_frame = aligned_frames.get_depth_frame()
#     color_frame = aligned_frames.get_color_frame()
#
#     if not aligned_depth_frame or not color_frame:
#         raise RuntimeError("未能获取对齐帧")
#
#     # 转换为 numpy 数组
#     depth_image = np.asanyarray(aligned_depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())
#
#     # 显示对齐图像
#     cv2.imshow("Color Image", color_image)
#     cv2.imshow("Aligned Depth Image", cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))
#     cv2.waitKey(1)
#
#     # 5. 生成点云
#     pc = rs.pointcloud()
#     points = pc.calculate(aligned_depth_frame)
#     pc.map_to(color_frame)
#
#     # 获取点云数据（顶点 + 颜色）
#     vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # XYZ
#     tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)  # UV
#
#     # 创建 open3d 点云对象
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(vtx)
#
#     # 提取颜色信息
#     # 获取颜色图像尺寸
#     h, w, _ = color_image.shape
#
#     # 获取点云纹理坐标（UV）
#     tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
#
#     # 构建颜色数组
#     colors = []
#     for u, v in tex:
#         # 注意：v 是行方向（y），u 是列方向（x），需乘以图像宽高并取整
#         x = min(max(int(u * w), 0), w - 1)
#         y = min(max(int(v * h), 0), h - 1)
#
#         rgb = color_image[y, x]  # BGR 顺序
#         colors.append(rgb / 255.0)  # 转为 [0,1] 范围
#
#     colors = np.asarray(colors)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#
#     # 可视化点云
#     o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
#
# finally:
#     pipeline.stop()
#     cv2.destroyAllWindows()






