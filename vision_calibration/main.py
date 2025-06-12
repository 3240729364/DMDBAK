import logging #用于记录程序运行时的日志信息
import numpy as np
import cv2 #OpenCV 是一个用于计算机视觉和图像处理的开源库
import pyrealsense2 as rs #深度相机的 Python 绑定库
import time

from log_setting import CommonLog
from config import CODE_fi, HOST_fi
from robotic_arm import Arm
import eye_in_hand_homogeneous_matrix

cam0_path = r'D:\PyCharm\Py_projects\vision_calibration\images\images'# 提前建立好的存储照片文件的目录

logger_ = logging.getLogger(__name__)#为当前模块创建一个与模块名称相关联的日志记录器
logger_ = CommonLog(logger_)

count = 1
run_loop = True


def displayD435():
    global pipeline
    pipeline = rs.pipeline()#创建一个 pipeline 对象，用于捕获和处理来自 RealSense 摄像头的数据
    config = rs.config()#这一行创建一个配置（config）对象，它将用于配置数据流参数
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 一行配置了数据流
    pipeline.start(config)

    global count

    try:
        while run_loop:
            frames = pipeline.wait_for_frames()#使用 pipeline 对象来等待并获取一帧数据
            color_frame = frames.get_color_frame()#获取的帧数据中提取彩色图像帧，如果没有可用的彩色图像帧，color_frame 将被设置为 None
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())#将彩色图像帧转换为 NumPy 数组，以便进一步的处理
            callback(color_image)



    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def callback(frame):
    # define picture to_down' coefficient of ratio
    scaling_factor = 1.0 #定义缩放比例
    global count
    global run_loop

    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    """当缩小图像时，使用INTER_AREA插值方式效果最好。当放大图像时，使用INTER_LINEAR和INTER_CUBIC效果最好，但是双三次插值法运算速度较慢，双线性插值法速度较快。"""
    cv2.imshow("Capture_Video", cv_img)  # 窗口显示，显示名为 Capture_Video

    if run_loop:
        k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        if k == ord('s'):  # 若检测到按键 ‘s’，打印字符串
            cv2.destroyWindow('Capture_Video')
            joint, pose_, error_code = arm.get_curr_arm_state()  # 获取当前机械臂状态
            logger_.info(f'获取状态：{"成功" if error_code == 0 else "失败"}，{f"当前位姿为{pose_}" if error_code == 0 else None} ')
            if error_code == 0:
                with open('images\poses.txt', 'w') as f:  # a+表示以追加模式打开.txt文件
                    pose_1 = [str(i) for i in pose_]
                    new_line = f'{",".join(pose_1)}\n'
                    f.write(new_line)
                    cv2.imwrite(cam0_path + str(count) + '.jpg', cv_img)
                    count += 1

                    pose_move = pose_
                    move_count = 1
                    for i in np.arange (-0.005,0.005,0.002):
                        pose_move[0] += i
                        error_code = arm.horizontal(pose_move)
                        logger_.info(f'第{move_count}次向x轴运动2cm：{"成功" if error_code == 0 else "失败"}，{f"当前位姿为{pose_move}" if error_code == 0 else None} ')
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        color_image = np.asanyarray(color_frame.get_data())
                        cv_img = cv2.resize(color_image, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
                        time.sleep(1)
                        if error_code ==0:
                            str_pose = [str(i) for i in pose_move]
                            new_line = f'{",".join(str_pose)}\n'
                            f.write(new_line)
                            cv2.imwrite(cam0_path + str(count) + '.jpg', cv_img)
                            count += 1
                            move_count += 1
                        else:
                            logger_.warning("机器人x方向移动失败：" + str(error_code))
                            break

                    pose_move = pose_
                    move_count = 1
                    for i in np.arange(-0.005, 0.005, 0.002):
                        pose_move[1] += i
                        error_code = arm.horizontal(pose_move)
                        logger_.info(
                            f'第{move_count}次向y轴运动2cm：{"成功" if error_code == 0 else "失败"}，{f"当前位姿为{pose_move}" if error_code == 0 else None} ')
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        color_image = np.asanyarray(color_frame.get_data())
                        cv_img = cv2.resize(color_image, None, fx=scaling_factor, fy=scaling_factor,
                                            interpolation=cv2.INTER_AREA)
                        time.sleep(1)
                        if error_code == 0:
                            str_pose = [str(i) for i in pose_move]
                            new_line = f'{",".join(str_pose)}\n'
                            f.write(new_line)
                            cv2.imwrite(cam0_path + str(count) + '.jpg', cv_img)
                            count += 1
                            move_count += 1
                        else:
                            logger_.warning("机器人y方向移动失败：" + str(error_code))
                            break

                    pose_move = pose_
                    move_count = 1
                    for i in np.arange(-0.005, 0.005, 0.002):
                        pose_move[2] += i
                        error_code = arm.horizontal(pose_move)
                        logger_.info(
                            f'第{move_count}次向z轴运动2cm：{"成功" if error_code == 0 else "失败"}，{f"当前位姿为{pose_move}" if error_code == 0 else None} ')
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        color_image = np.asanyarray(color_frame.get_data())
                        cv_img = cv2.resize(color_image, None, fx=scaling_factor, fy=scaling_factor,
                                            interpolation=cv2.INTER_AREA)
                        time.sleep(1)
                        if error_code == 0:
                            str_pose = [str(i) for i in pose_move]
                            new_line = f'{",".join(str_pose)}\n'
                            f.write(new_line)
                            cv2.imwrite(cam0_path + str(count) + '.jpg', cv_img)
                            count += 1
                            move_count += 1
                        else:
                            logger_.warning("机器人z方向移动失败：" + str(error_code))
                            break

                    pose_move = pose_
                    move_count = 1
                    for i in np.arange(-0.005, 0.005, 0.002):
                        pose_move[3] += i
                        error_code = arm.horizontal(pose_move)
                        logger_.info(
                            f'第{move_count}次绕x轴旋转0.002rad：{"成功" if error_code == 0 else "失败"}，{f"当前位姿为{pose_move}" if error_code == 0 else None} ')
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        color_image = np.asanyarray(color_frame.get_data())
                        cv_img = cv2.resize(color_image, None, fx=scaling_factor, fy=scaling_factor,
                                            interpolation=cv2.INTER_AREA)
                        time.sleep(1)
                        if error_code == 0:
                            str_pose = [str(i) for i in pose_move]
                            new_line = f'{",".join(str_pose)}\n'
                            f.write(new_line)
                            cv2.imwrite(cam0_path + str(count) + '.jpg', cv_img)
                            count += 1
                            move_count += 1
                        else:
                            logger_.warning("机器人绕x轴旋转失败：" + str(error_code))
                            break

            else:
                pass

            if count >=20 or error_code!=0 :
                run_loop = False


if __name__ == '__main__':
    # arm = Arm(CODE_fi, HOST_fi)
    # arm.change_frame()
    displayD435()
    # if count>=20:
    #     R, t = eye_in_hand_homogeneous_matrix.func()
    #     logger_.info(f"手眼标定完成,旋转矩阵为{R},平移矩阵为{t}")
    # else:
    #     logger_.warning("数据量过小，请增加数据量")
    # R, t = eye_in_hand_homogeneous_matrix.func()
    # logger_.info(f"手眼标定完成,旋转矩阵为{R},平移矩阵为{t}")