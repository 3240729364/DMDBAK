"""

眼在手上 计算得是 相机相对于机械臂末端 齐次变换矩阵
计算 这个矩阵需要得是  标定板相对于相机得次变换矩阵 * 相机相对于机械臂末端得齐次变换矩阵 * 机械臂末端相对于基座得齐次变换矩阵

机械臂末端相对于基座得齐次变换矩阵（也就是机械臂位姿变换得齐次变换矩阵）

"""
import csv
import numpy as np



def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz@Ry@Rx
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4) #构建一个4x4单位矩阵
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H

# # 示例
# pose_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 假设这是位姿列表中的一个元素
# H = pose_to_homogeneous_matrix(pose_list)
# print(H)



def save_matrices_to_csv(matrices, file_name):
    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))#目的是将所有输入矩阵按列连接在一起

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:#newline=''表示在写入时不插入额外的换行符
        csv_writer = csv.writer(csvfile)#创建一个CSV写入器 csv_writer
        for row in combined_matrix:
            csv_writer.writerow(row)


def poses_main(filepath):
    # 打开文本文件
    with open(filepath, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        lines = f.readlines()
    # 定义一个空列表，用于存储结果
    matrices = []
    # 遍历每一行数据
    lines = [float(i) for line in lines for i in line.split(',')]

    for i in range(0,len(lines),6):
        matrices.append(pose_to_homogeneous_matrix(lines[i:i+6]))


    # 将齐次变换矩阵列表存储到 CSV 文件中
    save_matrices_to_csv(matrices, f'RobotToolPose.csv')

def pose_convert(end_effector_pose):
    from scipy.spatial.transform import Rotation as R

    # 机械臂末端的位姿转换为齐次变换矩阵
    position = end_effector_pose[:3]
    orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
    """Rotation.from_euler 函数将欧拉角转换为旋转矩阵
    表示欧拉角是否以度为单位。如果设置为 True，则表示输入的欧拉角是以度为单位；如果设置为 False，则表示输入的欧拉角是以弧度为单位
    as_matrix() 将 Rotation 对象转换为一个3x3的旋转矩阵
    """
    print(f'orientation:{orientation}')

    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position
    return T_base_to_end_effector
if __name__ == "__main__":
    # 假设已经将位姿列表转换为齐次变换矩阵列表


    pose = [-0.079926997423172, 0.0061039999127388, 0.6547899842262268, 0.06199999898672104, 0.014000000432133675,
            -3.065000057220459]
    pose1 = [0,0,1,0,0,0]
    print(pose_convert(pose))
    print(pose_to_homogeneous_matrix(pose1))
