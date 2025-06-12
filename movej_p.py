# This Python file uses the following encoding: utf-8

# if __name__ == "__main__":
#     pass

# Python 画八字程序
import ctypes
import os
import sys
import time

if __name__ == "__main__":
    CUR_PATH = os.path.dirname(os.path.realpath(__file__))
    dllPath = os.path.join(CUR_PATH, "RM_Base.dll")
    pDll = ctypes.cdll.LoadLibrary(dllPath)

    #   API 初始化
    pDll.RM_API_Init(65, 0)

    #   连接机械臂
    byteIP = bytes("192.168.1.18", "gbk")
    nSocket = pDll.Arm_Socket_Start(byteIP, 8080, 200)
    print(nSocket)


    """
    这段代码涉及到使用 ctypes 模块将 Python 代码与动态链接库中的函数进行绑定，并控制机械臂进行 MoveJ 运动。
    前三行代码先定义了一个长度为6的浮点数数组类型(float_joint)，并定义了一个新的数组对象 (joint1)，表示机械臂六轴的位置信息。
    接下来的 pDll.Movej_Cmd.argtypes 和 pDll.Movej_Cmd.restype 表示将 Python 中的数据类型与动态链接库中的函数进行绑定。
    Movej_Cmd 是一个动态链接库中定义的函数，该函数的参数依次为 (int, float * 6, byte, float, bool)，
    对应的数据类型分别是 (ctypes.c_int, ctypes.c_float * 6, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)，
    其中 ctypes.c_int 表示整数类型，ctypes.c_float 表示浮点数类型，ctypes.c_byte 表示字节类型，ctypes.c_bool 表示布尔类型。
    最后一行代码启动 MoveJ 运动，将绑定好的数据传递给 Movej_Cmd 函数，并将其返回的值存储在 ret 中。
    MoveJ 运动是指机械臂以直线或者圆弧的形式从当前位置移动到目标点，所移动的路径取决于第三个参数，这里的参数设置为 20，表示直线轨迹。
    其中，nSocket 为之前建立的 socket 句柄，joint1 表示机械臂六轴的位置信息，
    20 表示 MoveJ 运动速度比例 1~100，即规划速度和加速度占关节最大线转速和加速度的比例为20，
    0 表示该运动的轨迹交融半径，默认为0，1 表示阻塞，等待机械臂到达位置或者规划失败。
    接下来进行 MoveJ 运动控制，使用 pDll.Movej_Cmd 将数据传递给动态链接库中的 C 函数，如果返回值不等于 0，则说明设置初始位置失败，
    直接输出提示信息，并使用 sys.exit() 退出程序。
    以上实现的是控制机械臂从当前位置直线移动到目标位置。
    """

    #   设置安装角度
    pDll.Set_Install_Pose.argtype = (ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool)
    nRet = pDll.Set_Install_Pose(nSocket, 0, 0, 0, 1)
    print("Set_Install_Pose ret:" + str(nRet))

    #   设置末端DH参数为标准版
    pDll.setLwt.argtype = ctypes.c_int
    pDll.setLwt(0)

    #   初始位置
    float_joint = ctypes.c_float * 6
    joint1 = float_joint()
    joint1[0] = 0
    joint1[1] = 0
    joint1[2] = 0
    joint1[3] = 0
    joint1[4] = 0
    joint1[5] = 0
    pDll.Movej_Cmd.argtypes = (ctypes.c_int, ctypes.c_float * 6, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
    pDll.Movej_Cmd.restype = ctypes.c_int
    nRet = pDll.Movej_Cmd(nSocket, joint1, 20, 0, 1)
    print("Movej_Cmd ret:" + str(nRet))
    time.sleep(1)
    if nRet != 0 :
        print("设置初始位置失败:" + str(nRet))
        sys.exit()

    #   画八字
    class POSE(ctypes.Structure):
        _fields_ = [("px", ctypes.c_float),
                    ("py", ctypes.c_float),
                    ("pz", ctypes.c_float),
                    ("rx", ctypes.c_float),
                    ("ry", ctypes.c_float),
                    ("rz", ctypes.c_float)]


    point_list = [[-0.342807, -0.092625, 0.33235, -1.806, 0.517, 1.923],  # 大椎
                  [-0.398911, -0.06661, 0.346175, -1.84, 0.568, 1.71],  #
                  [-0.45167, -0.049611, 0.357921, -2.062, 0.53, 1.316],  # 天宗

                  ]
    for i in range(len(point_list)):
        po1 = POSE()
        po1.px = point_list[i][0]
        po1.py = point_list[i][1]
        po1.pz = point_list[i][2]
        po1.rx = point_list[i][3]
        po1.ry = point_list[i][4]
        po1.rz = point_list[i][5]
        pDll.Movel_Cmd.argtypes = (ctypes.c_int, POSE, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        pDll.Movel_Cmd.restype = ctypes.c_int
        ret = pDll.Movel_Cmd(nSocket, po1, 10, 0, 1)
        if ret != 0:
            print("Movel_Cmd 1 失败:" + str(ret))

    float_joint = ctypes.c_float * 6
    joint1 = float_joint()
    joint1[0] = 0
    joint1[1] = 0
    joint1[2] = 0
    joint1[3] = 0
    joint1[4] = 0
    joint1[5] = 0
    ret = pDll.Movej_Cmd(nSocket, joint1, 10, 0, 1)

    i = 1
    while i < 5:
        time.sleep(1)
        i += 1

    #   关闭连接
    pDll.Arm_Socket_Close(nSocket)