import logging
import os

import supervisely_lib as sly

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import numpy as np
import tqdm


class UploadAPI:
    def __init__(self, project_id=None, ds_name='pointcloud'):
        self.api = sly.Api.from_env()

        self.project = self.api.project.get_info_by_id(project_id)
        self.dataset = self.api.dataset.get_info_by_name(project_id, ds_name)
        self.meta = self.api.project.get_meta(project_id)
        logger.info(f"Api works with project {self.project.id} dataset {self.dataset.name}")
        pclouds = self.api.pointcloud.get_list(self.dataset.id)
        self.pclouds_dict = {pc.name: pc for pc in pclouds}
        self.pcloud = None

    def download_cloud(self, cloud_name):
        self.pcloud = self.pclouds_dict[cloud_name]
        fpath = os.path.join(os.environ['DEBUG_APP_DIR'], self.pcloud.name)
        self.api.pointcloud.download_path(self.pcloud.id, fpath)
        return fpath

    def upload_annotation(self, annotation):
        self.api.pointcloud.annotation.append(self.pcloud.id, annotation)  # annotation upload
        logger.info(f'Annotation uploaded')

    def set_pcloud(self, cloud_name):
        self.pcloud = self.pclouds_dict[cloud_name]

    def download_dataset(self, dataset_dirname, split):
        obj_class_infos = self.api.object_class.get_list(self.project.id)
        obj_class_infos = {x.id: x.name for x in obj_class_infos}

        all_bboxes = []
        all_labels = []
        pcloud_paths = []

        save_path = os.path.join(os.environ['DEBUG_APP_DIR'], dataset_dirname, split)
        for k, v in tqdm.tqdm(self.pclouds_dict.items()):
            data = self.api.pointcloud.annotation.download(v.id)

            self.pcloud = self.pclouds_dict[v.name]
            pcloud_path = os.path.join(save_path, self.pcloud.name)
            self.api.pointcloud.download_path(self.pcloud.id, pcloud_path)

            pcloud_paths.append(pcloud_path)

            figures = data['figures']
            objects = data['objects']
            objects = {x['id']: x for x in objects}

            bboxes = []
            labels = []

            for figure in figures:
                geometry = figure['geometry']
                x, y, z = geometry['position']['x'], geometry['position']['y'], geometry['position']['z']
                w, l, h = geometry['dimensions']['x'], geometry['dimensions']['y'], geometry['dimensions']['z']
                yaw = geometry['rotation']['z']
                class_name = obj_class_infos[objects[figure['objectId']]['classId']]

                bboxes.append([x, y, z, w, l, h, yaw])
                labels.append(class_name)

            all_bboxes.append(bboxes)
            all_labels.append(labels)

        data = np.array([pcloud_paths, all_bboxes, all_labels], dtype=object)
        np.save(os.path.join(save_path, "dataset.npy"), data)
        logger.info(f"Dataset loaded to {save_path}")
