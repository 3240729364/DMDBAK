import os
import random
import tqdm
import shutil

"""
Segmentation of test set and validation set code
"""
Dataset_root = '../../backacupoint_data'
os.chdir(os.path.join(Dataset_root, 'labelme_jsons'))

test_frac = 0.2
random.seed(123)

folder = '.'
img_paths = os.listdir(folder)
random.shuffle(img_paths)

val_number = int(len(img_paths) * test_frac)
train_files = img_paths[val_number:]
val_files = img_paths[:val_number]

train_labelme_jsons_folder = 'train_labelme_jsons'
os.mkdir(train_labelme_jsons_folder)

for each in tqdm(train_files):
    src_path = os.path.join(folder, each)
    dst_path = os.path.join(train_labelme_jsons_folder, each)
    shutil.move(src_path, dst_path)

val_labelme_jsons_folder = 'val_labelme_jsons'
os.mkdir(val_labelme_jsons_folder)

for each in tqdm(val_files):
    src_path = os.path.join(folder, each)
    dst_path = os.path.join(val_labelme_jsons_folder, each)
    shutil.move(src_path, dst_path)

os.chdir('../../')