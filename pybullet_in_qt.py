"""
显示自定义界面，但RL环境仿真开始后会卡掉
只有机械臂的环境不会卡(单步仿真可行，或者无延时函数的循环，但只会显示最后一个动作)
"""
from math import pi

import untitled  # 注意这里，导入的是你.ui文件所生成的.py文件的名字


import pybullet as p
import pybullet_data
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class BulletWidget(QGLWidget):
    def __init__(self, parent=None):
        super(BulletWidget, self).__init__(parent)
        self.setMinimumSize(800, 600)

        self.physicsClient = p.connect(p.GUI)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        self.boxId = p.loadURDF(r"C:\Users\32407\Desktop\reinforcement learning\rm_65_6f_description\urdf\rm_65_6f_description.urdf", [0, 0, 0],useFixedBase=True)

        self.cameraTargetPosition = [0, 0, 1]

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateGL)
        self.timer.start(16)  # 更新周期为16毫秒，大约60帧每秒

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width/height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        p.stepSimulation()

        # 获取物体的位置和方向
        boxPos, _ = p.getBasePositionAndOrientation(self.boxId)
        # 计算相机位置，使其跟随物体移动
        cameraPos = [boxPos[0] - 3, boxPos[1] - 3, boxPos[2] + 3]
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.cameraTargetPosition, distance=2, yaw=0, pitch=-10, roll=180, upAxisIndex=2)
        aspect = self.width() / self.height()
        projectionMatrix = p.computeProjectionMatrixFOV(fov=60, aspect=aspect, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=self.width(), height=self.height(), viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        glDrawPixels(self.width(), self.height(), GL_RGBA, GL_UNSIGNED_BYTE, px)

class MainWindow(QtWidgets.QMainWindow, untitled.Ui_MainWindow, QGLWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # 传递自己
        # self.central_widget = QtWidgets.QWidget()
        # self.setCentralWidget(self.central_widget)
        # self.layout = QVBoxLayout(self.central_widget)
        # self.bullet_widget = BulletWidget(self.central_widget)
        # self.layout.addWidget(self.bullet_widget)
        # self.setWindowTitle("PyBullet with PyQt")

        self.flag =False
        self.init()  # 构建init方法

        # 通过init方法集中绑定槽函数
    def init(self):
        self.pushButton.clicked.connect(self.open_camera)  # 绑定打开相机槽函数open_camera
        self.pushButton_2.clicked.connect(self.close_camera)  # 绑定关闭相机槽函数close_camera
        self.pushButton_3.clicked.connect(self.start_simulation)
        self.pushButton_4.clicked.connect(self.set_joint)


    def open_camera(self):
        self.flag = True
        print(self.flag)

        self.Layout = QtWidgets.QVBoxLayout(self.openGLWidget)
        self.bullet_widget = BulletWidget(self.openGLWidget)
        self.Layout.addWidget(self.bullet_widget)
        self.setWindowTitle("PyBullet with PyQt")


    def start_simulation(self):
        neutral_angle = [0, 0, 0, 0, 0, 0]
        neutral_angle = [x * pi / 180 for x in neutral_angle]
        p.setJointMotorControlArray(window.bullet_widget.boxId, [0, 1, 2, 3, 4, 5], p.POSITION_CONTROL,
                                    # 忽略关节6（固定关节）
                                    targetPositions=neutral_angle)
        p.stepSimulation()
        # time.sleep(2)
        # neutral_angle = [0, 0, 0, 0, 90, 0]
        # neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        # p.setJointMotorControlArray(window.bullet_widget.boxId, [0, 1, 2, 3, 4, 5], p.POSITION_CONTROL,
        #                             # 忽略关节6（固定关节）
        #                             targetPositions=neutral_angle)


    def set_joint(self):
        angle = [0, 90, 0, 90, 90, 0]
        angle = [x * pi / 180 for x in angle]
        p.setJointMotorControlArray(window.bullet_widget.boxId, [0, 1, 2, 3, 4, 5], p.POSITION_CONTROL,
                                    # 忽略关节6（固定关节）
                                    targetPositions=angle)
        p.stepSimulation()


    def close_camera(self):  # 关闭环境(摄像头)
        if self.flag == True:  # 如果仿真打开，则关闭仿真环境
            self.bullet_widget.Env.close()
        self.close() # 关闭窗口






if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 800, 600)
    window.show()

    sys.exit(app.exec_())
