"""
还是不行

"""
import time
from datetime import datetime

import numpy as np
import pybullet as p

import pybullet_data
import math

physicsClient = p.connect(p.GUI)
p.resetSimulation()
p.setTimeStep(1/240)  # 设置仿真间隔，例如 1/240 秒
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -10)

boxId = p.loadURDF(
    r"C:\Users\32407\Desktop\reinforcement learning\rm_65_6f_description\urdf\rm_65_6f_description.urdf", [0, 0,0],
    useFixedBase=True)
dt = datetime.now()
t = (dt.second / 60.) * 2. * math.pi
p.setRealTimeSimulation(1)  #通过使用setrealtimessimulation命令让物理服务器根据其实时时钟(RTC)自动步进仿真，可以实时运行仿真。如果您启用实时模拟，则不需要调用“stepSimulation”

kukaEndEffectorIndex = 5
#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]



p.setGravity(0, 0, -10)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1

useOrientation = 1
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1

ikSolver = 0

#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15


from sympy import symbols, solve, atan2


def arc_traj(points, theta5, T, step):
    q = np.zeros((step, 6))  # 角度
    t = np.zeros((step, 1))  # 时间

    # 求圆心和半径
    _x0, _y0, _z0 = symbols('_x0 _y0 _z0')
    p1, p2, p3 = points
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    b = (x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1)
    c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    d = a * x1 + b * y1 + c * z1
    eq1 = a * _x0 + b * _y0 + c * _z0 - d
    eq2 = (_x0 - 0.5 * (x1 + x2)) * (x2 - x1) + (_y0 - 0.5 * (y1 + y2)) * (y2 - y1) + (_z0 - 0.5 * (z1 + z2)) * (z2 - z1)
    eq3 = (_x0 - 0.5 * (x2 + x3)) * (x3 - x2) + (_y0 - 0.5 * (y2 + y3)) * (y3 - y2) + (_z0 - 0.5 * (z2 + z3)) * (z3 - z2)
    a1 = solve((eq1, eq2, eq3), (_x0, _y0, _z0))
    print(a1)
    x0, y0, z0 = np.int32(a1[_x0]), np.int32(a1[_y0]), np.int32(a1[_z0])
    p0 = np.array([x0, y0, z0])
    print(p0)
    R = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
    print(R)

    # 求齐次变换矩阵
    W = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
    U = (p1 - p0) / R
    V = np.cross(W, U)

    U = U.reshape((3, 1))  # 或者 U = U[:, np.newaxis]
    V = V.reshape((3, 1))  # 或者 V = V[:, np.newaxis]
    W = W.reshape((3, 1))  # 或者 W = W[:, np.newaxis]
    transT = np.vstack((np.hstack((U, V, W, p0.reshape((3, 1)))), np.array([0, 0, 0, 1])))  # 原坐标系看新建坐标系
    print(transT)
    # 计算轨迹并转换
    pointT = []
    for i in np.linspace(0, np.arctan2(p3[1], p3[0]), step ):
        pointT.append(np.dot(transT , np.array([R * np.cos(i), R * np.sin(i), 0, 1])))
    pointT = np.array(pointT)[:,:3]
    print("轨迹点集合：\n",pointT)



    # 角度变化??
    def myikine2(points, theta5, step):
        jointPoses=np.zeros((step,6))

        for i in range(1):
            pos = points[i]   #  [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
            # end effector points down, not up (in case useOrientation==1)
            orn = p.getQuaternionFromEuler([0, -math.pi, 0])

            if (useNullSpace == 1):
                if (useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(boxId, kukaEndEffectorIndex, pos, orn, ll, ul,
                                                              jr, rp)
                else:
                    jointPoses = p.calculateInverseKinematics(boxId,
                                                              kukaEndEffectorIndex,
                                                              pos,
                                                              lowerLimits=ll,
                                                              upperLimits=ul,
                                                              jointRanges=jr,
                                                              restPoses=rp)
            else:
                if (useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(boxId,
                                                              kukaEndEffectorIndex,
                                                              pos,
                                                              orn,
                                                              jointDamping=jd,
                                                              solver=ikSolver,
                                                              maxNumIterations=100,
                                                              residualThreshold=.01)
                else:
                    jointPoses = p.calculateInverseKinematics(boxId,
                                                              kukaEndEffectorIndex,
                                                              pos,
                                                              solver=ikSolver)

        return jointPoses



    q[:step, :] = myikine2(pointT, theta5[:2], step)#pointT[:step, :]
    # q[step:2*step, :] = myikine2(pointT[step:2*step, :], theta5[1:], step)
    t[:step] = np.linspace(T[0], T[1], step).reshape((step, 1))
    # t[step:2*step] = np.linspace(T[1], T[2], step).reshape((step, 1))

    return q, t, pointT





# 示例调用
# 调用函数计算轨迹的关节角度和时间
points = np.array([[0.2, 0, 0], [0, 0.2, 0], [-0.2, 0, 0]])  # 三个点的坐标
theta5 = np.array([0, 0, 0])  # 初始角度
T = np.array([0, 1, 2])  # 时间段
step = 100  # 步数

q, t,pointT  = arc_traj(points, theta5, T, step)
# print("q的数组形状", q.shape)
# for i in range(20):
#     print("关节角度：\n", q[i,:])
#     pos = p.getLinkState(boxId, 5)
#     print("末端坐标：\n",pos)

# # 在环境中绘制pointT的全部点位
# def draw_points(pointT):
#     for i in range(pointT.shape[0]):
#         p.addUserDebugPoints(pointT[i, :], pointColorsRGB=[1, 0, 0], pointSize=0.01)
# draw_points(pointT)
#
# print("时间：\n", t)


# 模拟画半圆测试  --未成功
pos= p.getLinkState(boxId, 5)  # 6是机械臂的最后一个关节的索引

positions=[pos[0]]
for i in range(q.shape[0]):


    # 设置关节角度
    angle = q[i,:]

    p.setJointMotorControlArray(boxId, [0, 1, 2, 3, 4, 5], p.POSITION_CONTROL,targetPositions=angle)


    print("关节角度：\n", q[i,:])
    pos = p.getLinkState(boxId, 5)
    print("末端坐标：\n",pos[0])

    pos= p.getLinkState(boxId, 5)  # 6是机械臂的最后一个关节的索引
    pos=pos[0]

    # 在PyBullet中绘制轨迹
    if i > 0:
        # p.addUserDebugLine(prevPose, pointT, [0, 0, 0.3], 1, trailDuration)
        p.addUserDebugLine((positions[i - 1][0], positions[i - 1][1], positions[i - 1][2]),
                           (pos[0], pos[1], pos[2]),
                           lineColorRGB=[1, 0, 0])
    positions.append(pos)

time.sleep(100)
p.disconnect()


