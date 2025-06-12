import os
import sys

import mne
# %matplotlib inline
import numpy as np
masterPath= r'E:\pycharm\Py_Projects\learn_pytorch\EEG_motor_imagery\FBCNetToolbox\codes\\'
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
from saveData import parseBci42aFile

# Mention the file path to the dataset
GDF_filename = r"E:\pycharm\Py_Projects\learn_pytorch\EEG_motor_imagery\FBCNetToolbox\data\bci42a\originalData\A01E.gdf"
EDF_filename = r"C:\Users\32407\Documents\WeChat Files\wxid_f9hj622ggjn422\FileStorage\File\2024-07\ERP1-20240705151628-filter.edf"

GDF_raw = mne.io.read_raw_gdf(GDF_filename)
EDF_raw = mne.io.read_raw_edf(EDF_filename)

print(GDF_raw.info)
print(GDF_raw._annotations)

print(EDF_raw.info)
print(EDF_raw._annotations)

