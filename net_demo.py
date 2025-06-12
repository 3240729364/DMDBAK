import pyrealsense2 as rs #深度相机的 Python 绑定库
import numpy as np
import cv2
# from predict import predict_single_person
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
import json
import matplotlib.pyplot as plt
import os
import ctypes
import sys
import time

cam0_path = r'D:\PyCharm\Py_projects\A01_HRNet\real-time_images\images'
count = 1
run_loop = True


def displayD435():
    global pipeline
    pipeline = rs.pipeline()#创建一个 pipeline 对象，用于捕获和处理来自 RealSense 摄像头的数据
    config = rs.config()#这一行创建一个配置（config）对象，它将用于配置数据流参数
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)#一行配置了数据流
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
    global run_loop

    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    """当缩小图像时，使用INTER_AREA插值方式效果最好。当放大图像时，使用INTER_LINEAR和INTER_CUBIC效果最好，但是双三次插值法运算速度较慢，双线性插值法速度较快。"""
    cv2.imshow("Capture_Video", cv_img)  # 窗口显示，显示名为 Capture_Video

    if run_loop:
        k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        if k == ord('s'):  # 若检测到按键 ‘s’，打印字符串
            cv2.destroyWindow('Capture_Video')
            cv2.imwrite(cam0_path + str(count) + '.jpg', cv_img)
            run_loop = False


point_name = ["dazhui",
        "left_jianjing",
        "right_jianjing",
        "left_naoshu",
        "right_naoshu",
        "left_jianzhen",
        "right_jianzhen",
        "left_dazhu",
        "right_dazhu",
        "left_fengmen",
        "right_fengmen",
        "left_feishu",
        "right_feishu",
        "left_jueyinshu",
        "right_jueyinshu",
        "left_xinshu",
        "right_xinshu",
        "left_gaohuang",
        "right_gaohuang",
        "left_tianzong",
        "right_tianzong",
        "left_geshu",
        "right_geshu",
        "left_ganshu",
        "right_ganshu",
        "left_danshu",
        "right_danshu",
        "left_pishu",
        "right_pishu",
        "left_weishu",
        "right_weishu",
        "left_sanjiaoshu",
        "right_sanjiaoshu",
        "left_shenshu",
        "right_shenshu",
        "left_dachangshu",
        "right_dachangshu"]

point_color = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
               (240, 2, 127), (240, 2, 127),
               (255, 255, 51), (255, 255, 51),
               (254, 153, 41), (44, 127, 184),
               (217, 95, 14), (0, 0, 255),
               (255, 255, 51), (255, 255, 51), (228, 26, 28),
               (49, 163, 84), (252, 176, 243), (0, 176, 240),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (128, 0, 128), (255, 165, 0), (0, 255, 0),
               (255, 192, 203), (128, 128, 128), (0, 255, 255),
               (192, 192, 192), (255, 105, 180), (0, 128, 128),
               (0, 0, 128), (255, 69, 0), (75, 0, 130),
               (255, 215, 0), (0, 0, 139), (186, 85, 211),
               (144, 238, 144), (128, 0, 0), (255, 20, 147)
               ]

def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 2,
                   draw_text: bool = True,
                   font: str = 'arial.ttf',
                   font_size: int = 80):
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i],
                         outline=(255, 255, 255))
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

    return img


class_list= {
    'supercategory': 'back',
    'id': 1,
    'name': 'back',
    'keypoints': ['dazhui', 'jianjing', 'naoshu', 'jianzhen', 'dazhu', 'fengmen', 'feishu', 'jueyinshu', 'xinshu', 'gaohuang', 'tianzong', 'geshu', 'ganshu', 'danshu', 'pishu', 'weishu', 'sanjiaoshu', 'shenshu', 'dachangshu'], # 大小写敏感
}
def process_single_json(labelme_json, image_id=1):
    '''
    输入labelme的json数据，输出coco格式的每个框的关键点标注信息
    '''
    with open(labelme_json, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    global ANN_ID

    coco_annotations = []

    for each_ann in labelme['shapes']:  # 遍历该json文件中的所有标注

        if each_ann['shape_type'] == 'rectangle':  # 筛选出框

            # 该框元数据
            bbox_dict = {}
            bbox_dict['category_id'] = 1
            bbox_dict['segmentation'] = []

            bbox_dict['iscrowd'] = 0
            bbox_dict['segmentation'] = []
            bbox_dict['image_id'] = image_id
            bbox_dict['id'] = ANN_ID
            # print(ANN_ID)
            ANN_ID += 1

            # 获取该框坐标
            bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_w = bbox_right_bottom_x - bbox_left_top_x
            bbox_h = bbox_right_bottom_y - bbox_left_top_y
            bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]  # 左上角x、y、框的w、h
            bbox_dict['area'] = bbox_w * bbox_h

            # # 筛选出分割多段线
            # for each_ann in labelme['shapes']:  # 遍历所有标注
            #     if each_ann['shape_type'] == 'polygon':  # 筛选出分割多段线标注
            #         # 第一个点的坐标
            #         first_x = each_ann['points'][0][0]
            #         first_y = each_ann['points'][0][1]
            #         if (first_x > bbox_left_top_x) & (first_x < bbox_right_bottom_x) & (
            #                 first_y < bbox_right_bottom_y) & (first_y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
            #             bbox_dict['segmentation'] = list(
            #                 map(lambda x: list(map(lambda y: round(y, 2), x)), each_ann['points']))  # 坐标保留两位小数
            #             # bbox_dict['segmentation'] = each_ann['points']

            # 筛选出该个体框中的所有关键点
            bbox_keypoints_dict = {}
            for each_ann in labelme['shapes']:  # 遍历所有标注

                if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                    # 关键点横纵坐标
                    x = int(each_ann['points'][0][0])
                    y = int(each_ann['points'][0][1])
                    label = each_ann['label']
                    if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & (y < bbox_right_bottom_y) & (
                            y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                        if label not in bbox_keypoints_dict:
                            bbox_keypoints_dict[label] = [[x, y]]
                        else:
                            bbox_keypoints_dict[label].append([x, y])

            bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
            # print(bbox_keypoints_dict)

            """先添加大椎一个穴位、再添加其他穴位"""
            bbox_dict['keypoints'] = []
            for each_class in class_list['keypoints']:
                if each_class == 'dazhui':
                    if each_class in bbox_keypoints_dict:
                        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][0])
                        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][1])
                        bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
                    else:
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)

            for each_class in class_list['keypoints']:
                if each_class != 'dazhui' :
                    if each_class in bbox_keypoints_dict:
                        if bbox_keypoints_dict[each_class][0][0] <= bbox_keypoints_dict[each_class][1][0]:
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][0])
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][1])
                            bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1][0])
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1][1])
                            bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
                        else:
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1][0])
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1][1])
                            bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][0])
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][1])
                            bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
                    else:
                        # with open('./omission_val.txt', 'a') as file:
                        #     output = f"{labelme['imagePath']} {each_class} 遗漏\n"
                        #     print(labelme['imagePath'], each_class, '遗漏')
                        #     file.write(output)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)

            coco_annotations.append(bbox_dict)

    return coco_annotations


def aijiu_move():
    global Velocity
    Velocity=10
    for num in range(2):

        pDll.Movej_Cmd.argtypes = (
        ctypes.c_int, ctypes.c_float * 6, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)  # 设置函数参数类型
        pDll.Movej_Cmd.restype = ctypes.c_int  # 函数返回类型

        # if ret != 0:
        #     print("设置初始位置失败:" + str(ret))
        #     sys.exit()

        joint_point_list = [-45.719,5.623,124.16,-98.695,-89.957,-14.562]# 大椎


        float_joint = ctypes.c_float * 6
        joint1 = float_joint()


        joint1[0] = joint_point_list[0]
        joint1[1] = joint_point_list[1]
        joint1[2] = joint_point_list[2]
        joint1[3] = joint_point_list[3]
        joint1[4] = joint_point_list[4]
        joint1[5] = joint_point_list[5]
        ret = pDll.Movej_Cmd(nSocket, joint1, Velocity, 0, 1)

        time.sleep(1)

        class POSE(ctypes.Structure):
            _fields_ = [("px", ctypes.c_float),
                        ("py", ctypes.c_float),
                        ("pz", ctypes.c_float),
                        ("rx", ctypes.c_float),
                        ("ry", ctypes.c_float),
                        ("rz", ctypes.c_float)]

        point_list=[[-0.5376,0.0924,0.1098,-2.628,1.15,0.164],  # 左 肩井
                    [-0.5931,-0.04518,0.11299,-1.891,1.239,1.119],  #左 臑俞
                    [-0.59860, -0.0755, 0.1515, -1.827, 1.11, 1.466],  # 左 肩贞
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
            ret = pDll.Movel_Cmd(nSocket, po1, Velocity, 0, 1)
            if ret != 0:
                print("Movel_Cmd 1 失败:" + str(ret))
                sys.exit()


    for num in range(2):


        class POSE(ctypes.Structure):
            _fields_ = [("px", ctypes.c_float),
                        ("py", ctypes.c_float),
                        ("pz", ctypes.c_float),
                        ("rx", ctypes.c_float),
                        ("ry", ctypes.c_float),
                        ("rz", ctypes.c_float)]

        point_list=[[-0.302851,-0.017085,0.3515,-1.805,0.543,2.234],# 肺腧
                    [-0.284859,0.03372,0.349238,-1.703,0.704,2.044],
                    [-0.283055, 0.04819, 0.351503, -1.709, 0.69, 1.972],
                    [-0.28807, 0.088354, 0.353495, -1.713, 0.82, 1.727],
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
            ret = pDll.Movel_Cmd(nSocket, po1, Velocity, 0, 1)
            if ret != 0:
                print("Movel_Cmd 1 失败:" + str(ret))
                sys.exit()

    for num in range(2):


        class POSE(ctypes.Structure):
            _fields_ = [("px", ctypes.c_float),
                        ("py", ctypes.c_float),
                        ("pz", ctypes.c_float),
                        ("rx", ctypes.c_float),
                        ("ry", ctypes.c_float),
                        ("rz", ctypes.c_float)]

        point_list=[[-0.284551,0.143562,0.353778,-1.693,0.786,1.648],# 肝俞
                    [-0.264452,0.200818,0.342906,-1.555,0.707,1.645],# 脾俞
                    [-0.271315, 0.234348, 0.345133, -1.552, 0.763, 1.562],#
                    [-0.2714, 0.270803, 0.341175, -1.524, 0.828, 1.417],
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
            ret = pDll.Movel_Cmd(nSocket, po1, Velocity, 0, 1)
            if ret != 0:
                print("Movel_Cmd 1 失败:" + str(ret))
                sys.exit()
    for num in range(2):


        class POSE(ctypes.Structure):
            _fields_ = [("px", ctypes.c_float),
                        ("py", ctypes.c_float),
                        ("pz", ctypes.c_float),
                        ("rx", ctypes.c_float),
                        ("ry", ctypes.c_float),
                        ("rz", ctypes.c_float)]

        point_list=[[-0.335658,0.277665,0.350277,-1.678,0.746,1.299],# 大肠俞
                    [-0.34101,0.21919,0.351584,-1.693,0.793,1.523],# 脾俞
                    [-0.342827, 0.18218, 0.350035, -1.662, 0.821, 1.696],#
                    [-0.342827, 0.139499, 0.350035, -1.662, 0.821, 1.696],
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
            ret = pDll.Movel_Cmd(nSocket, po1, Velocity, 0, 1)
            if ret != 0:
                print("Movel_Cmd 1 失败:" + str(ret))
                sys.exit()
    for num in range(2):


        class POSE(ctypes.Structure):
            _fields_ = [("px", ctypes.c_float),
                        ("py", ctypes.c_float),
                        ("pz", ctypes.c_float),
                        ("rx", ctypes.c_float),
                        ("ry", ctypes.c_float),
                        ("rz", ctypes.c_float)]

        point_list=[[-0.342827, 0.099408, 0.350035, -1.662, 0.821, 1.696],# 膈腧
                    [-0.342827, 0.045518, 0.355035, -1.662, 0.821, 1.696],# 心腧
                    [-0.344756, -0.043475, 0.343178, -1.709, 0.602, 1.825],#
                    [-0.342807, -0.092625, 0.33035, -1.806, 0.517, 1.923],
                    [-0.342807, -0.092625, 0.35035, -1.806, 0.517, 1.923],
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
            ret = pDll.Movel_Cmd(nSocket, po1, Velocity, 0, 1)
            if ret != 0:
                print("Movel_Cmd 1 失败:" + str(ret))
                sys.exit()


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


if __name__ == '__main__':
    CUR_PATH = os.path.dirname(os.path.realpath(__file__))
    dllPath = os.path.join(CUR_PATH, "RM_Base.dll")
    pDll = ctypes.cdll.LoadLibrary(dllPath)

    #   API 初始化
    pDll.RM_API_Init(65, 0)

    #   连接机械臂
    byteIP = bytes("192.168.1.18", "gbk")
    nSocket = pDll.Arm_Socket_Start(byteIP, 8080, 200)
    print(nSocket)

    float_joint = ctypes.c_float * 6
    joint1 = float_joint()
    joint1[0] = 0
    joint1[1] = 0
    joint1[2] = 0
    joint1[3] = 0
    joint1[4] = 0
    joint1[5] = 0
    ret = pDll.Movej_Cmd(nSocket, joint1, 10, 0, 1)

    # 拍照位姿
    float_joint = ctypes.c_float * 6
    joint1 = float_joint()
    joint1[0] = 22.675
    joint1[1] = 20.676
    joint1[2] = 60.32
    joint1[3] = -86.19
    joint1[4] = 23.71
    joint1[5] = 55.25
    ret = pDll.Movej_Cmd(nSocket, joint1, 10, 0, 1)

    # 准备按摩
    # float_joint = ctypes.c_float * 6
    # joint1 = float_joint()
    # joint1[0] = -26.953
    # joint1[1] = -8.865
    # joint1[2] = 121.303
    # joint1[3] = -71.915
    # joint1[4] = -66.401
    # joint1[5] = -0.881
    # ret = pDll.Movej_Cmd(nSocket, joint1, 10, 0, 1)

    displayD435()
    os.chdir(os.path.dirname(os.getcwd()))
    # predict_single_person(img_path="./real-time_images/images1.jpg", r=30, font_size=80)

    # os.chdir("./demo")
    ANN_ID = 0
    coco_annotations = process_single_json(labelme_json=r"E:\pycharm\Py_Projects\learn_pytorch\other\beibu_Color.json")
    keypoints = coco_annotations[0]['keypoints']
    keypoints = [item for index, item in enumerate(keypoints) if (index + 1) % 3 != 0]
    keypoints = np.array(keypoints).reshape((37, 2))
    img = cv2.imread("_MG_6713.JPG")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plot_img = draw_keypoints(img, keypoints=keypoints, thresh=0.2, r=30, font_size = 80)
    plt.imshow(plot_img)
    plt.show()
    plot_img.save("test_result.jpg")
    #
    # aijiu_move()
