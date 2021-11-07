
dataset_type = "SuperviselyDataset"
data_root = '/data/slyproject'
class_names = ['Car', 'Pedestrian', 'Cyclist', 'DontCare']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)

file_client_args = dict(backend='disk')
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'train/dataset.npy',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=class_names,
#     sample_groups=dict(Car=1, Pedestrian=1, Cyclist=1),
#     points_loader=dict(
#         type='LoadPointsFromSlyFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=[0, 1, 2, 3],
#         file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromSlyFile'),
    dict(
        type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    #dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromSlyFile')]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromSlyFile'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root)),
    val=dict(
        type=dataset_type,
        data_root=data_root),
    test=dict(
        type=dataset_type,
        data_root=data_root))

evaluation = dict(interval=1, pipeline=eval_pipeline)
