# import pygame  # 导入 Pygame 库，用于创建游戏窗口和处理事件
# from pygame.locals import *  # 导入 Pygame 的本地模块，包含常用的变量和函数
#
# from OpenGL.GL import *  # 导入 OpenGL 的核心功能
# from OpenGL.GLUT import *  # 导入 OpenGL 的实用工具库
# from OpenGL.GLU import *  # 导入 OpenGL 的实用工具库
#
from STL_loader import *
#
# # 定义三角形的顶点
# vertices = [
#     [0, 1, 0],  # 顶点0
#     [-1, -1, 0],  # 顶点1
#     [1, -1, 0]  # 顶点2
# ]
#
# # 定义三角形的颜色
# colors = [
#     [1, 0, 0],  # 红色
#     [0, 1, 0],  # 绿色
#     [0, 0, 1]  # 蓝色
# ]
#
# def Triangle():
#     """
#     绘制三角形
#     """
#     glBegin(GL_TRIANGLES)  # 开始绘制三角形
#     for i, vertex in enumerate(vertices):
#         glColor3fv(colors[i])  # 设置颜色
#         glVertex3fv(vertex)  # 设置顶点
#     glEnd()  # 结束绘制三角形
#
# def main():
#     """
#     主函数
#     """
#     pygame.init()  # 初始化 Pygame
#     display = (800, 600)
#     pygame.display.set_mode(display, DOUBLEBUF|OPENGL)  # 创建窗口
#
#     gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)  # 设置透视参数
#     glTranslatef(0.0, 0.0, -5)  # 平移视图
#
#     while True:  # 主循环
#         for event in pygame.event.get():  # 处理事件
#             if event.type == pygame.QUIT:  # 如果是退出事件，则退出程序
#                 pygame.quit()
#                 quit()
#
#         glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)  # 清除屏幕和深度缓冲
#
#         # Triangle()  # 绘制三角形
#         model0 = loader('RobotSimulator-master/STLFile/Link1.STL')
#         model0.draw()
#
#         pygame.display.flip()  # 刷新屏幕
#         pygame.time.wait(10)  # 稍微等待一下，减少 CPU 占用
#
# main()  # 调用主函数，启动程序

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

vertices2 = np.array([
    [[0.2, 0.2, 0.2], [-0.2, 0.2, 0.2], [-0.2, -0.2, 0.2], [0.2, -0.2, 0.2]],  # 前
    [[0.2, 0.2, -0.2], [0.2, -0.2, -0.2], [-0.2, -0.2, -0.2], [-0.2, 0.2, -0.2]],  # 后
    [[0.2, 0.2, 0.2], [0.2, 0.2, -0.2], [-0.2, 0.2, -0.2], [-0.2, 0.2, 0.2]],  # 左
    [[0.2, -0.2, 0.2], [0.2, -0.2, -0.2], [-0.2, -0.2, -0.2], [-0.2, -0.2, 0.2]],  # 右
    [[0.2, 0.2, 0.2], [0.2, -0.2, 0.2], [0.2, -0.2, -0.2], [0.2, 0.2, -0.2]],  # 上
    [[-0.2, 0.2, 0.2], [-0.2, -0.2, 0.2], [-0.2, -0.2, -0.2], [-0.2, 0.2, -0.2]]  # 下
])

colours = np.array([
    [1, 0, 1], [1, 0, 1],
    [1, 1, 1], [1, 1, 1],
    [0, 1, 1], [0, 1, 1]
])
IS_PERSPECTIVE = True  # 透视投影
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 0.5, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
LEFT_IS_DOWNED = False
CameraPos = np.array([0.0, 0.0, 2])
CameraFront = np.array([0, 0, 0])
CameraUp = np.array([0, 1, 0])
SCALE_K = np.array([1, 1, 1])
yaw = 0
pitch = 0
MOUSE_X, MOUSE_Y = 0, 0
WIN_W = 640
WIN_H = 480


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）


def show():
    global IS_PERSPECTIVE, VIEW
    global CameraPos, CameraFront, CameraUp
    global SCALE_K
    global WIN_W, WIN_H
    global vertices2

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    if IS_PERSPECTIVE:
        glFrustum(VIEW[0], VIEW[1], VIEW[2], VIEW[3], VIEW[4], VIEW[5])

    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 几何变换
    glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])

    # 视点
    gluLookAt(
        CameraPos[0], CameraPos[1], CameraPos[2],
        CameraFront[0], CameraFront[1], CameraFront[2],
        CameraUp[0], CameraUp[1], CameraUp[2]
    )

    glViewport(0, 0, WIN_W, WIN_H)

    glBegin(GL_LINES)

    # 以红色绘制x轴
    glColor3f(1.0, 0.0, 0.0)  # 设置当前颜色为红色不透明
    glVertex3f(-0.5, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
    glVertex3f(0.5, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）

    # 以绿色绘制y轴
    glColor3f(0.0, 1.0, 0.0)  # 设置当前颜色为绿色不透明
    glVertex3f(0.0, -0.5, 0.0)  # 设置y轴顶点（y轴负方向）
    glVertex3f(0.0, 0.5, 0.0)  # 设置y轴顶点（y轴正方向）

    # 以蓝色绘制z轴
    glColor3f(0.0, 0.0, 1.0)  # 设置当前颜色为蓝色不透明
    glVertex3f(0.0, 0.0, -0.5)  # 设置z轴顶点（z轴负方向）
    glVertex3f(0.0, 0.0, 0.5)  # 设置z轴顶点（z轴正方向）

    glEnd()

    # 显示正方体
    # for i in range(vertices2.shape[0]):
    #     glBegin(GL_QUADS)
    #     points = vertices2[i, :]
    #     color = colours[i, :]
    #     for point in points:
    #         glColor3f(color[0], color[1], color[0])
    #         glVertex3f(point[0], point[1], point[2])
    #     glEnd()

    # 显示模型
    model0 = loader('../RobotSimulator-master/STLFile/Link1.STL')
    model0.draw()

    glutSwapBuffers()


def Mouse_click(button, state, x, y):
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y
    global SCALE_K

    MOUSE_X = x
    MOUSE_Y = y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state == GLUT_DOWN


def Mouse_motion(x, y):
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y
    global yaw, pitch
    global CameraPos

    if LEFT_IS_DOWNED:
        dx = x - MOUSE_X
        dy = y - MOUSE_Y
        MOUSE_X = x
        MOUSE_Y = y

        sensitivity = 0.2
        dx = dx * sensitivity
        dy = dy * sensitivity

        yaw = yaw + dx
        pitch = pitch + dy

        if pitch > 89:
            pitch = 89
        if pitch < -89:
            pitch = -89

        CameraPos[0] = np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
        CameraPos[1] = np.sin(np.radians(pitch))
        CameraPos[2] = np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))

        glutPostRedisplay()


if __name__ == '__main__':
    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)
    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    glutCreateWindow("OpenGL")

    init()
    glutDisplayFunc(show)
    glutMouseFunc(Mouse_click)
    glutMotionFunc(Mouse_motion)
    glutMainLoop()

