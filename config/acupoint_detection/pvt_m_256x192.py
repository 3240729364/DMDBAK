_base_ = ['mmpose::_base_/default_runtime.py']

resume = True
dataset_type = 'CocoDataset'
data_mode = 'bottomup'
data_root = '../../backacupoint_data/'

dataset_info = {
    'dataset_name':'backacupoint_data',
    'classes':('back'),
    'paper_info':{
        'author':'back acupoint from HFUT and AHUCM',
        'title':'Backacupoint Keypoints Detection',
        'container':'HFUT',
        'year':'2024',
        'homepage':''
    },
    'keypoint_info':{
        0: {'name': 'dazhui', 'id': 0, 'color': (240, 2, 127), 'type': '', 'swap': ''},
        1: {'name': 'left_jianjing', 'id': 1, 'color': (255, 255, 51), 'type': '', 'swap': 'right_jianjing'},
        2: {'name': 'right_jianjing', 'id': 2, 'color': (255, 255, 51), 'type': '', 'swap': 'left_jianjing'},
        3: {'name': 'left_naoshu', 'id': 3, 'color': (0, 0, 255), 'type': '', 'swap': 'right_naoshu'},
        4: {'name': 'right_naoshu', 'id': 4, 'color': (0, 0, 255), 'type': '', 'swap': 'left_naoshu'},
        5: {'name': 'left_jianzhen', 'id': 5, 'color': (0, 255, 0), 'type': '', 'swap': 'right_jianzhen'},
        6: {'name': 'right_jianzhen', 'id': 6, 'color': (0, 255, 0), 'type': '', 'swap': 'left_jianzhen'},
        7: {'name': 'left_dazhu', 'id': 7, 'color': (228, 26, 28), 'type': '', 'swap': 'right_dazhu'},
        8: {'name': 'right_dazhu', 'id': 8, 'color': (228, 26, 28), 'type': '', 'swap': 'left_dazhu'},
        9: {'name': 'left_fengmen', 'id': 9, 'color': (128, 0, 128), 'type': '', 'swap': 'right_fengmen'},
        10: {'name': 'right_fengmen', 'id': 10, 'color': (128, 0, 128), 'type': '', 'swap': 'left_fengmen'},
        11: {'name': 'left_feishu', 'id': 11, 'color': (255, 165, 0), 'type': '', 'swap': 'right_feishu'},
        12: {'name': 'right_feishu', 'id': 12, 'color': (255, 165, 0), 'type': '', 'swap': 'left_feishu'},
        13: {'name': 'left_jueyinshu', 'id': 13, 'color': (0, 255, 255), 'type': '', 'swap': 'right_jueyinshu'},
        14: {'name': 'right_jueyinshu', 'id': 14, 'color': (0, 255, 255), 'type': '', 'swap': 'left_jueyinshu'},
        15: {'name': 'left_xinshu', 'id': 15, 'color': (255, 192, 203), 'type': '', 'swap': 'right_xinshu'},
        16: {'name': 'right_xinshu', 'id': 16, 'color': (255, 192, 203), 'type': '', 'swap': 'left_xinshu'},
        17: {'name': 'left_gaohuang', 'id': 17, 'color': (128, 128, 128), 'type': '', 'swap': 'right_gaohuang'},
        18: {'name': 'right_gaohuang', 'id': 18, 'color': (128, 128, 128), 'type': '', 'swap': 'left_gaohuang'},
        19: {'name': 'left_tianzong', 'id': 19, 'color': (255, 105, 180), 'type': '', 'swap': 'right_tianzong'},
        20: {'name': 'right_tianzong', 'id': 20, 'color': (255, 105, 180), 'type': '', 'swap': 'left_tianzong'},
        21: {'name': 'left_geshu', 'id': 21, 'color': (192, 192, 192), 'type': '', 'swap': 'right_geshu'},
        22: {'name': 'right_geshu', 'id': 22, 'color': (192, 192, 192), 'type': '', 'swap': 'left_geshu'},
        23: {'name': 'left_ganshu', 'id': 23, 'color': (75, 0, 130), 'type': '', 'swap': 'right_ganshu'},
        24: {'name': 'right_ganshu', 'id': 24, 'color': (75, 0, 130), 'type': '', 'swap': 'left_ganshu'},
        25: {'name': 'left_danshu', 'id': 25, 'color': (255, 69, 0), 'type': '', 'swap': 'right_danshu'},
        26: {'name': 'right_danshu', 'id': 26, 'color': (255, 69, 0), 'type': '', 'swap': 'left_danshu'},
        27: {'name': 'left_pishu', 'id': 27, 'color': (186, 85, 211), 'type': '', 'swap': 'right_pishu'},
        28: {'name': 'right_pishu', 'id': 28, 'color': (186, 85, 211), 'type': '', 'swap': 'left_pishu'},
        29: {'name': 'left_weishu', 'id': 29, 'color': (144, 238, 144), 'type': '', 'swap': 'right_weishu'},
        30: {'name': 'right_weishu', 'id': 30, 'color': (144, 238, 144), 'type': '', 'swap': 'left_weishu'},
        31: {'name': 'left_sanjiaoshu', 'id': 31, 'color': (128, 0, 0), 'type': '', 'swap': 'right_sanjiaoshu'},
        32: {'name': 'right_sanjiaoshu', 'id': 32, 'color': (128, 0, 0), 'type': '', 'swap': 'left_sanjiaoshu'},
        33: {'name': 'left_shenshu', 'id': 33, 'color': (255, 20, 147), 'type': '', 'swap': 'right_shenshu'},
        34: {'name': 'right_shenshu', 'id': 34, 'color': (255, 20, 147), 'type': '', 'swap': 'left_shenshu'},
        35: {'name': 'left_dachangshu', 'id': 35, 'color': (169, 209, 142), 'type': '', 'swap': 'right_dachangshu'},
        36: {'name': 'right_dachangshu', 'id': 36, 'color': (169, 209, 142), 'type': '', 'swap': 'left_dachangshu'},
    },
    'skeleton_info': {
    }
}

NUM_KEYPOINTS = len(dataset_info['keypoint_info'])
dataset_info['joint_weights'] = [1.55, 1.39, 1.39, 1.12, 1.12, 0.93, 0.93, 1.59, 1.59, 1.57, 1.57, 1.32, 1.32, 1.12, 1.12,
                                 0.95, 0.95, 0.48, 0.48, 0.51, 0.51, 1.49, 1.49, 1.22, 1.22, 1.61, 1.61, 1.77, 1.77, 2.02, 2.02,
                                 2.17, 2.17, 2.0, 2.0, 2.0, 2.0]
dataset_info['sigmas'] = [0.010832657099758445, 0.01479784096640981, 0.01479784096640981, 0.020035903605798922,
                                        0.020035903605798922, 0.025481829975504553, 0.025481829975504553, 0.012634637580605627,
                                        0.012634637580605627, 0.01389128821891222, 0.01389128821891222, 0.017402795678105768,
                                        0.017402795678105768, 0.021116402297445543, 0.021116402297445543, 0.025915938659046456,
                                        0.025915938659046456, 0.07226917724635369, 0.07226917724635369, 0.05116838560067946,
                                        0.05116838560067946, 0.018442355944627033, 0.018442355944627033, 0.0232143109127412,
                                        0.0232143109127412, 0.019138551135633054, 0.019138551135633054, 0.018214099699745273,
                                        0.018214099699745273, 0.016410553831964585, 0.016410553831964585, 0.015798815813137638,
                                        0.015798815813137638, 0.016560461101169995, 0.016560461101169995, 0.014560172552594177,
                                        0.014560172552594177]

max_epochs = 210
val_interval = 5
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 64
val_batch_size = 32
stage2_num_epochs = 0
base_lr = 4e-3
randomness = dict(seed=21)

optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='PyramidVisionTransformer',
        num_layers=[3, 4, 18, 2],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/whai362/PVT/'
            'releases/download/v2/pvt_medium.pth'),
    ),
    neck=dict(type='FeatureMapProcessor', select_index=3),
    head=dict(
        type='HeatmapHead',
        in_channels=512,
        out_channels=NUM_KEYPOINTS,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='train_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='val_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

default_hooks = {
    'checkpoint': {'save_best': 'coco/AP','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]

# evaluators
val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'val_coco.json')
]

test_evaluator = val_evaluator


