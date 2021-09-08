_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_kitti.py',
    '../_base_/datasets/supervisely_3class.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# dataset settings
data_root = '/data/slyproject'
class_names = ['Pedestrian', 'Cyclist', 'Car']
# PointPillars adopted a different sampling strategies among classes

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type = 'LIDAR', load_dim = 4, use_dim = 4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [

    dict(type='LoadPointsFromFile', coord_type = 'LIDAR', load_dim = 4, use_dim = 4),
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
runner = dict(max_epochs=2)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=2)
