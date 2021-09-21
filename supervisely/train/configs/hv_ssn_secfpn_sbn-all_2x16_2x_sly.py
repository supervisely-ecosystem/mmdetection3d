_base_ = [
    '/mmdetection3d/configs/_base_/models/hv_pointpillars_fpn_lyft.py',
    '/mmdetection3d/configs/_base_/default_runtime.py',
]

dataset_type = "SuperviselyDataset"
data_root = '/data/slyproject'

# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 23])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
checkpoint_config = dict(interval=20)

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

point_cloud_range = [-100, -100, -5, 100, 100, 3]
# Note that the order of class names should be consistent with
# the following anchors' order
class_names = ['car', 'truck']

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline,
               classes=class_names,
               type=dataset_type,
               data_root=data_root),
    val=dict(pipeline=test_pipeline,
             classes=class_names,
             type=dataset_type,
             data_root=data_root),
    test=dict(pipeline=test_pipeline,
             classes=class_names,
             type=dataset_type,
             data_root=data_root))


# model settings
model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        feat_channels=[32, 64],
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(output_shape=[800, 800]),
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        _delete_=True,
        type='ShapeAwareHead',
        num_classes=9,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGeneratorPerCls',
            ranges=[
                    [-100, -100, -1.0715024, 100, 100, -1.0715024],
                    [-100, -100, -0.3033737, 100, 100, -0.3033737]],
            sizes=[
                [1.92, 4.75, 1.71],  # car
                [2.84, 10.24, 3.44]  # truck
            ],
            custom_values=[],
            rotations=[0, 1.57],
            reshape_out=False),
        tasks=[
            dict(
                num_class=2,
                class_names=['car', 'truck'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(1, 1),
                norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01))
        ],
        assign_per_class=True,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        pts=dict(
            assigner=[
                dict(  # car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(  # truck
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)))


