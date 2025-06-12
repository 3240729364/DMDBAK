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


np.set_printoptions(precision=8,suppress=True)#precision打印浮点数时的精度，suppress禁用科学计数法

iamges_path = r'C:\Users\34830\Desktop\realman\d345\data_collection_d435_win\images' #手眼标定采集的标定版图片所在路径
file_path = r'C:\Users\34830\Desktop\realman\d345\data_collection_d435_win\images\poses.txt'  #采集标定板图片时对应的机械臂末端的位姿 从第一行到最后一行 需要和采集的标定板的图片顺序进行对应


def func():

    path = os.path.dirname(__file__)

    # 角点的个数以及棋盘格间距
    XX = 10 #标定板的中长度对应的角点的个数
    YY = 6  #标定板的中宽度对应的角点的个数
    L = 0.025#标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    # cv2.TERM_CRITERIA_EPS - 如果满足了指定准确度，epsilon就停止算法迭代
    # cv2.TERM_CRITERIA_MAX_ITER - 在指定次数的迭代后就停止算法
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L*objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点

    size = None


    for i in range(0, 50):   #标定好的图片在iamges_path路径下，从0.jpg到x.jpg   一次采集的图片最多不超过50张，我们遍历从0.jpg到50.jpg ，选择能够读取的到的图片

        image = f"{iamges_path}\images{i}.jpg"

        if os.path.exists(image):

            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1] #获取灰度图像尺寸
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            """在灰度图像中查找棋盘格的角点 
            ret是一个标志，指示是否找到了角点,corners包含找到的角点的坐标
            None，表示没有特殊的角点搜索标志被使用"""

            if ret:

                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)




    N = len(img_points)

    # 标定,得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    """None: 相机的内部参数（摄像机矩阵 mtx 和畸变系数 dist）的初始估计值。在这里，设置为 None 表示使用默认的初始值。
        None: 相机的外部参数（旋转向量 rvecs 和平移向量 tvecs）的初始估计值。同样，设置为 None 表示使用默认的初始值。
        ret: 标定的平均误差。它是一个标量，表示标定过程的平均重投影误差。"""

    # print("ret:", ret)
    print("内参矩阵:\n", mtx) # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("-----------------------------------------------------")

    poses_main(file_path)
    # 机器人末端在基座标系下的位姿

    tool_pose = np.loadtxt(f'{path}/RobotToolPose.csv',delimiter=',')#delimiter=',': 指定 CSV 文件中的列之间的分隔符
    R_tool = []
    t_tool = []
    for i in range(int(N)):
        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3])

    R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    print(R)
    print(t)
    return R,t



# 旋转矩阵
# rotation_matrix ,translation_vector = func()
# print(rotation_matrix,translation_vector)
#
# if __name__ == '__main__':
#
#     # 将旋转矩阵转换为四元数
#     rotation = R.from_matrix(rotation_matrix)
#     quaternion = rotation.as_quat()
#
#     qw, qx, qy, qz = quaternion
#     x, y, z = translation_vector.flatten()
#
#     print(f"qw: {qw}\nqx: {qx}\nqy: {qy}\nqz: {qz}\nx: {x}\ny: {y}\nz: {z}")