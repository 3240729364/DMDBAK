import json
import os


def process_single_json(labelme, image_id=1):
    '''
    Input labelme's json data, output coco format keypoint labeling information for each box
    '''

    global ANN_ID

    coco_annotations = []

    for each_ann in labelme['shapes']:

        if each_ann['shape_type'] == 'rectangle':

            bbox_dict = {}
            bbox_dict['category_id'] = 1
            bbox_dict['segmentation'] = []

            bbox_dict['iscrowd'] = 0
            bbox_dict['segmentation'] = []
            bbox_dict['image_id'] = image_id
            bbox_dict['id'] = ANN_ID
            ANN_ID += 1

            bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_w = bbox_right_bottom_x - bbox_left_top_x
            bbox_h = bbox_right_bottom_y - bbox_left_top_y
            bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]
            bbox_dict['area'] = bbox_w * bbox_h

            bbox_keypoints_dict = {}
            for each_ann in labelme['shapes']:

                if each_ann['shape_type'] == 'point':
                    x = int(each_ann['points'][0][0])
                    y = int(each_ann['points'][0][1])
                    label = each_ann['label']
                    if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & (y < bbox_right_bottom_y) & (
                            y > bbox_left_top_y):
                        if label not in bbox_keypoints_dict:
                            bbox_keypoints_dict[label] = [[x, y]]
                        else:
                            bbox_keypoints_dict[label].append([x, y])

            bbox_dict['num_keypoints'] = 2 * len(bbox_keypoints_dict) - 1

            """First add a point labeling of the Dazui, then add other points labeling"""
            bbox_dict['keypoints'] = []
            for each_class in class_list['keypoints']:
                if each_class == 'dazhui':
                    if each_class in bbox_keypoints_dict:
                        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][0])
                        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][1])
                        bbox_dict['keypoints'].append(2)
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
                            bbox_dict['keypoints'].append(2)
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1][0])
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1][1])
                            bbox_dict['keypoints'].append(2)
                        else:
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1][0])
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1][1])
                            bbox_dict['keypoints'].append(2)
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][0])
                            bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0][1])
                            bbox_dict['keypoints'].append(2)
                    else:
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)

            coco_annotations.append(bbox_dict)

    return coco_annotations


def process_folder():
    IMG_ID = 0

    for labelme_json in os.listdir():


        if labelme_json.split('.')[-1] == 'json':

            with open(labelme_json, 'r', encoding='utf-8') as f:
                labelme = json.load(f)

                img_dict = {}
                img_dict['file_name'] = labelme['imagePath']
                img_dict['height'] = labelme['imageHeight']
                img_dict['width'] = labelme['imageWidth']
                img_dict['id'] = IMG_ID
                coco['images'].append(img_dict)

                coco_annotations = process_single_json(labelme, image_id=IMG_ID)
                coco['annotations'] += coco_annotations

                IMG_ID += 1

                print(labelme_json, 'Done.')

        else:
            pass

coco = {}
coco['info'] = {}
coco['info']['description'] = 'back acupoint from HFUT and AHUCM'
coco['info']['year'] = 2024
coco['info']['date_created'] = '2024/07/19'

class_list= {
    'supercategory': 'back',
    'id': 1,
    'name': 'back',
    'keypoints': ['dazhui', 'jianjing', 'naoshu', 'jianzhen', 'dazhu', 'fengmen', 'feishu', 'jueyinshu', 'xinshu', 'gaohuang', 'tianzong', 'geshu', 'ganshu', 'danshu', 'pishu', 'weishu', 'sanjiaoshu', 'shenshu', 'dachangshu'], # 大小写敏感
}

coco['categories'] = []
coco['categories'].append(class_list)

coco['images'] = []
coco['annotations'] = []
IMG_ID = 0
ANN_ID = 0

"Processing of val data"
Dataset_root = '../../backacupoint_data'
path = os.path.join(Dataset_root, 'labelme_jsons', 'val_labelme_jsons')
os.chdir(path)
process_folder()

coco_path = '../../val_coco.json'
with open(coco_path, 'w') as f:
     json.dump(coco, f, indent=2)
