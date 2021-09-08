import numpy as np
from os import path as osp

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet3d.datasets import Custom3DDataset
import supervisely_lib as sly


@DATASETS.register_module()
class SuperviselyDataset(Custom3DDataset):
    CLASSES = None
    def __init__(self, data_root, ann_file="", pipeline=None, classes=None, modality=None, box_type_3d="LIDAR", filter_empty_gt=True, test_mode=False):
        self.project_fs = sly.PointcloudProject.read_single(data_root)
        self.meta = self.project_fs.meta
        SuperviselyDataset.CLASSES = [x.name for x in self.meta.obj_classes]  # TODO: compare with config classes!

        for dataset_fs in self.project_fs:
            self.dataset = dataset_fs
            self.items = list(self.dataset._item_to_ann)
            break

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)



    def load_annotations(self, ann_file):
        return self.items

    def read_annotation(self, ann_path):
        ann_json = sly.io.json.load_json_file(ann_path)
        ann = sly.PointcloudAnnotation.from_json(ann_json, self.meta)
        objects = []
        bboxes = []
        labels = []

        if ann.figures:
            for fig in ann.figures:
                geometry = fig.geometry

                bbox = [
                    geometry.position.x, geometry.position.y, geometry.position.z,
                    geometry.dimensions.x, geometry.dimensions.z, geometry.dimensions.z,
                    geometry.rotation.z
                ]

                bboxes.append(np.array(bbox, dtype=np.float32))
                labels.append(SuperviselyDataset.CLASSES.index(fig.parent_object.obj_class.name))

        return np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.long)

    def get_ann_info(self, index):
        item = self.items[index]
        item_path, related_images_dir, ann_path = self.dataset.get_item_paths(item)
        gt_bboxes_3d, gt_labels_3d = self.read_annotation(ann_path)

        if not gt_bboxes_3d.any():
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5))

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d)

        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        item = self.data_infos[index]
        item_path, _, _ = self.dataset.get_item_paths(item)

        input_dict = dict(
            pts_filename=item_path,
            sample_idx=0,
            file_name=item_path)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            #if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
            #    return None
        return input_dict

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        #if self.filter_empty_gt and \
        #        (example is None or
        #         ~(example['gt_labels_3d'] != -1).any()):
        #    return None
        print(example)
        return example