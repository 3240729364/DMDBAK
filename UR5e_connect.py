
'''
1.连接网线
2.手动设置以太网的地址：192.168.1.x   掩码：255.255.255.0
3.显示到连接器网络后，在控制器中启用远程控制
4.发送命令连接机械臂

'''
import math
import socket
import struct

import numpy as np

dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d', 'I target': '6d',
       'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
       'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
       'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
       'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
       'Tool Accelerometer values': '3d',
       'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd',
       'V main': 'd',
       'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
       'Elbow position': '3d', 'Elbow velocity': '3d'}



HOST = "192.168.1.100"  # The remote host
PORT = 30003  # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.connect((HOST, PORT))
    print("连接成功")
except TimeoutError as e:
    print(f"连接超时: {e}")

data = s.recv(1108)
# data = s.recv(1108)
names = []
ii = range(len(dic))
for key, i in zip(dic, ii):
    fmtsize = struct.calcsize(dic[key])
    data1, data = data[0:fmtsize], data[fmtsize:]
    fmt = "!" + dic[key]
    names.append(struct.unpack(fmt, data1))
    dic[key] = dic[key], struct.unpack(fmt, data1)
print(names)
print(dic)


# channels:
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
#   - defaults
# show_channel_urls: true
# ssl_verify: false

a=dic["q actual"]
a2=np.array(a[1])
print(a2*180/math.pi)

# tcp_commend = 'get_actual_joint_positions()\n'
# tcp_commend += 'get_actual_tcp_pose()\n'

# strL = "movel(p[0.2,0.3,0.5,0,0,3.14],a=0.5,v=0.25)\n"

# s.send(str.encode(tcp_commend))


# 关闭连接
s.close()