#!/usr/bin/env python3
# !coding=utf-8
import logging

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

from config import CODE_fi, HOST_fi
from log_setting import CommonLog
from robotic_arm import Arm



logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

cam0_path = 'images'  #存储采集到的标定板的图片的文件夹路径
poses_path = 'poses.txt' #存放采集到的机械臂末端位姿的文件路径

def callback(data):

    global count, bridge


    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")

    cv2.imshow("Capture_Video", cv_img)  # 窗口显示，显示名为 Capture_Test

    k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
    if k == ord('s'):  # 若检测到按键 ‘s’，打印字符串
        joint, pose_, error_code = arm.get_curr_arm_state()  # 获取当前机械臂状态
        logger_.info(f'获取状态：{"成功" if error_code == 0 else "失败"},{f"当前位姿为{pose_}" if error_code == 0 else None} ')
        if error_code == 0:
            with open(poses_path, 'a+') as f:
                # 将列表中的元素用空格连接成一行
                pose_ = [str(i) for i in pose_]
                new_line = f'{",".join(pose_)}\n'
                # 将新行附加到文件的末尾
                f.write(new_line)


            cv2.imwrite(cam0_path + str(count) + '.jpg', cv_img)  # 保存；

            count += 1

    else:
        pass


def displayWebcam():


    rospy.init_node('webcam_display', anonymous=True)

    # make a video_object and init the video object
    global count, bridge
    count = 0
    bridge = CvBridge()
    rospy.Subscriber('/stereo_inertial_publisher/color/image', Image, callback)
    rospy.spin()


if __name__ == '__main__':

    arm = Arm(CODE_fi, HOST_fi)
    arm.change_frame()

    displayWebcam()
