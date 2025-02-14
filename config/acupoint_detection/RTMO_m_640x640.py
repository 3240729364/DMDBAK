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

train_cfg = dict(max_epochs=800, val_interval=5, dynamic_intervals=[(580, 1)])
train_batch_size = 16
val_batch_size = 16
stage2_num_epochs = 0
base_lr = 4e-3
randomness = dict(seed=21)

optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({'neck.encoder': dict(lr_mult=0.05)})),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=5,
        T_max=280,
        end=280,
        by_epoch=True,
        convert_to_iter_based=True),
    # this scheduler is used to increase the lr from 2e-4 to 5e-4
    dict(type='ConstantLR', by_epoch=True, factor=2.5, begin=280, end=281),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=281,
        T_max=300,
        end=580,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=580, end=600),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# codec settings
input_size = (640, 640)
metafile = 'configs/_base_/datasets/back.py'
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

widen_factor = 0.75
deepen_factor = 0.67

model = dict(
    type='BottomupPoseEstimator',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1),
        ]),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/'
            'pretrained_models/yolox_m_8x8_300e_coco_20230829.pth',
            prefix='backbone.',
        )),
    neck=dict(
        type='HybridEncoder',
        in_channels=[192, 384, 768],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[1, 2],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU'))),
        projector=dict(
            type='ChannelMapper',
            in_channels=[256, 256],
            kernel_size=1,
            out_channels=384,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=2)),
    head=dict(
        type='RTMOHead',
        num_keypoints=NUM_KEYPOINTS,
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=256,
            cls_feat_channels=256,
            channels_per_group=36,
            pose_vec_channels=384,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish')),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='oks',
            oks_calculator=dict(type='PoseOKS', metainfo=metafile)),
        prior_generator=dict(
            type='MlvlPointGenerator',
            centralize_points=True,
            strides=[16, 32]),
        dcc_cfg=dict(
            in_channels=384,
            feat_channels=128,
            num_bins=(192, 256),
            spe_channels=128,
            gau_cfg=dict(
                s=128,
                expansion_factor=2,
                dropout_rate=0.0,
                drop_path=0.0,
                act_fn='SiLU',
                pos_enc='add')),
        overlaps_power=0.5,
        loss_cls=dict(
            type='VariFocalLoss',
            reduction='sum',
            use_target_weight=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo=metafile,
            loss_weight=30.0),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mle=dict(
            type='MLECCLoss',
            use_target_weight=True,
            loss_weight=1e-2,
        ),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    test_cfg=dict(
        input_size=input_size,
        score_thr=0.1,
        nms_thr=0.65,
    ))

backend_args = dict(backend='local')

# pipelines
train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type='YOLOXMixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        data_mode=data_mode,
        ann_file='train_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline_stage1,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='val_coco.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

default_hooks = {
    'checkpoint': {'save_best': 'coco/AP','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

custom_hooks = [
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=20,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(
        type='RTMOModeSwitchHook',
        epoch_attributes={
            280: {
                'proxy_target_cc': True,
                'loss_mle.loss_weight': 5.0,
                'loss_oks.loss_weight': 10.0
            },
        },
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_coco.json',
    score_mode='bbox',
    nms_mode='none',
)

test_evaluator = val_evaluator


