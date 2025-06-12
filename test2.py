import numpy as np
import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import move_circle
# clid = p.connect(p.SHARED_MEMORY)
# if (clid < 0):
p.connect(p.GUI)
  #p.connect(p.SHARED_MEMORY_GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf", [0, 0, -0.3])
kukaId = p.loadURDF(r"C:\Users\32407\Desktop\reinforcement learning\rm_65_6f_description\urdf\rm_65_6f_description.urdf", [0, 0, 0],useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 5
numJoints = p.getNumJoints(kukaId)
if (numJoints != 6):
  exit()

#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#resetposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
  p.resetJointState(kukaId, i, rp[i])

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
useRealTimeSimulation = 0
ikSolver = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15


i=0
while 1:
  i+=1
  #p.getCameraImage(320,
  #                 200,
  #                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  #                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
  if (useRealTimeSimulation):
    dt = datetime.now()
    t = (dt.second / 60.) * 2. * math.pi
  else:
    t = t + 0.01

  if (useSimulation and useRealTimeSimulation == 0):
    p.stepSimulation()

  # # 计算轨迹点集合
  # points = np.array([[0.2, 0, 0], [0, 0.2, 0], [-0.2, 0, 0]])  # 三个点的坐标
  # theta5 = np.array([0, 0, 0])  # 初始角度
  # T = np.array([0, 1, 2])  # 时间段
  # step = 100  # 步数
  # q, t,pointT=move_circle.arc_traj(points, theta5, T, step)
  for i in range(1):
    # pos = pointT[t*100]
    #end effector points down, not up (in case useOrientation==1)
    pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul,
                                                  jr, rp)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  lowerLimits=ll,
                                                  upperLimits=ul,
                                                  jointRanges=jr,
                                                  restPoses=rp)
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  orn,
                                                  jointDamping=jd,
                                                  solver=ikSolver,
                                                  maxNumIterations=100,
                                                  residualThreshold=.01)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  solver=ikSolver)

    if (useSimulation):
      for i in range(numJoints):
        p.setJointMotorControl2(bodyIndex=kukaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(numJoints):
        p.resetJointState(kukaId, i, jointPoses[i])

  ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
  if (hasPrevPose):
    p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
    p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
  prevPose = pos
  prevPose1 = ls[4]
  hasPrevPose = 1

p.disconnect()

# 建立目标点，一个小圆球
# collisionTargetId = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
#                                                 radius=0.02)
# target = p.createMultiBody(baseMass=0,  # 质量
#                                      baseCollisionShapeIndex=collisionTargetId,
#                                      basePosition=[1, 0, 0])
# pos= p.getLinkState(boxId, 5)  # 6是机械臂的最后一个关节的索引
# print(pos[0])




# 等待退出连接
print("OK")
time.sleep(6) # 等待6s
p.disconnect()

# print(np.array((6,5,8,4)))
# print(np.array((6,5,8,4))[2])

