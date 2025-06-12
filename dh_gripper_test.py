import time

from pyDHgripper import AG95

gripper = AG95(port='COM4')

gripper.set_force(100)  # 20-100
gripper.set_vel(1000)   # 0-1000
gripper.set_pos(100)  # 0-1000



print(gripper.read_state())