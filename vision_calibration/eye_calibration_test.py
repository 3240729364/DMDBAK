# coding=utf-8
# copied by ysh in 2021/12/08
"""
眼在手外 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os.path
import cv2
import numpy as np

from save_poses import poses_main

from scipy.spatial.transform import Rotation as R


np.set_printoptions(precision=8,suppress=True)

# iamges_path = r'C:\Users\hr\Desktop\D435\data_collection_d435_win\images' #手眼标定采集的标定版图片所在路径
# file_path = r'C:\Users\hr\Desktop\D435\data_collection_d435_win\images\poses.txt'  #采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应


def func():

    path = os.path.dirname(__file__)

    # 角点的个数以及棋盘格间距
    XX = 10 #标定板的中长度对应的角点的个数
    YY = 6  #标定板的中宽度对应的角点的个数
    L = 0.025 #标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L*objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点


    for i in range(18):   #标定好的图片在iamges_path路径下，从0.jpg到x.jpg   一次采集的图片最多不超过50张，我们遍历从0.jpg到50.jpg ，选择能够读取的到的图片

        # image = f"iamges_path\\{i}.jpg"
        t = str(i + 1)
        iamges_path = r"C:\Users\34830\Desktop\realman\d345\data_collection_d435_win\images"  # 拍照文件目录
        image = f"{iamges_path}\images{i}.jpg"
        ChessImg = cv2.imread(image,0)  # 左视图
        # print(ChessImg)
        # gray = cv2.cvtColor(ChessImg, cv2.COLOR_BGR2GRAY)
        # if os.path.exists(image):
        #
        #     img = cv2.imread(image)
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        #     size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(ChessImg, (XX, YY), None)

        if ret:

            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(ChessImg, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)




    N = len(img_points)

    # 标定,得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, ChessImg.shape[::-1], None, None)
    R_all_chess_to_cam_1 = []
    # T_all_chess_to_cam_1 = []
    # T_all_chess_to_cam_1.append(tvecs)
    rvecs_list = []
    # L_t_matrices = []
    M_RT = []
    for r_vector in rvecs:
        r_matrix, _ = cv2.Rodrigues(r_vector)
        R_all_chess_to_cam_1.append(r_matrix)
    # for i in range(30):
    #     lt_matrix = np.concatenate((rvecs_list[i], tvecs[i]), axis=1)
    #     T_all_chess_to_cam_1.append(lt_matrix)
    # for lt_matrix in L_t_matrices:
    #     four_d_matrixL = np.vstack((lt_matrix, [0, 0, 0, 1]))
    #     M_RT.append(four_d_matrixL)
    # R_all_chess_to_cam_1.append(M_RT[i][:3, :3])
    # T_all_chess_to_cam_1.append(M_RT[i][:3, 3].reshape((3, 1)))
    # print("ret:", ret)
    print("内参矩阵:\n", mtx) # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("-----------------------------------------------------")
    file_path = r'C:\Users\34830\Desktop\realman\d345\data_collection_d435_win\images\poses.txt'  # 采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应

    poses_main(file_path)
    # 机器人末端在基座标系下的位姿

    tool_pose = np.loadtxt(f'{path}/RobotToolPose.csv',delimiter=',')
    R_tool = []
    t_tool = []
    for i in range(int(N)):
        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3])

    R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    print(R)
    print(t)
    for i in range(18):
        # a = i + 1
        RT_end_to_base = np.column_stack((R_tool[i], t_tool[i]))
        RT_end_to_base = np.row_stack((RT_end_to_base, np.array([0, 0, 0, 1])))
        # print(RT_end_to_base)

        RT_chess_to_cam = np.column_stack((R_all_chess_to_cam_1[i], tvecs[i]))
        RT_chess_to_cam = np.row_stack((RT_chess_to_cam, np.array([0, 0, 0, 1])))
        # print(RT_chess_to_cam)

        RT_cam_to_end = np.column_stack((R, t))
        RT_cam_to_end = np.row_stack((RT_cam_to_end, np.array([0, 0, 0, 1])))
        # print(RT_cam_to_end)

        RT_chess_to_base = RT_end_to_base @ RT_cam_to_end @ RT_chess_to_cam  # 即为固定的棋盘格相对于机器人基坐标系位姿
        RT_chess_to_base = np.linalg.inv(RT_chess_to_base)
        print('第', i, '次')
        # print(RT_chess_to_base[:3, :])
        print(RT_chess_to_base)
        print('')
    return R,t



# 旋转矩阵
rotation_matrix ,translation_vector = func()

if __name__ == '__main__':

    # 将旋转矩阵转换为四元数
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()

    qw, qx, qy, qz = quaternion
    x, y, z = translation_vector.flatten()

    print(f"qw: {qw}\nqx: {qx}\nqy: {qy}\nqz: {qz}\nx: {x}\ny: {y}\nz: {z}")