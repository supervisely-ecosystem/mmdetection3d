from dotenv import dotenv_values

config = {
    **dotenv_values(".env"),  # load shared development variables
    **dotenv_values(".env.secret")  # load sensitive variables
}

import os
for k,v in config.items():
    os.environ[k] = v


import logging
import os

import numpy as np
import supervisely_lib as sly
from supervisely_lib.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

from inference import predict, decode_prediction

logger = logging.getLogger()


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


class Annotation:
    @staticmethod
    def pred_to_sly_geometry(labels, reverse=True):
        geometry = []

        for l in labels:
            x, y, z, dx, dy, dz, heading = l
            position = Vector3d(float(x), float(y), float(z))

            if reverse:
                yaw = float(heading) - np.pi
                yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi
            else:
                yaw = -heading

            rotation = Vector3d(0, 0, float(yaw))
            dimension = Vector3d(float(dx), float(dy), float(dz))
            g = Cuboid3d(position, rotation, dimension)
            geometry.append(g)
        return geometry

    @staticmethod
    def _collect_meta(labels, geometry=Cuboid3d):
        """
        :param labels: list of red KITTI labels
        :return: sly.ProjectMeta
        """
        unique_labels = np.unique(labels)
        obj_classes = [sly.ObjClass(k, geometry) for k in unique_labels]
        meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        return meta

    @staticmethod
    def create_annotation(bboxes, labels):
        geometry_list = Annotation.pred_to_sly_geometry(bboxes)
        figures = []
        objs = []

        meta = Annotation._collect_meta(labels=labels)

        for label, geometry in zip(labels, geometry_list):  # by object in point cloud
            pcobj = sly.PointcloudObject(meta.get_obj_class(label))
            figures.append(sly.PointcloudFigure(pcobj, geometry))
            objs.append(pcobj)

        pc_annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(objs), figures)
        return pc_annotation


if __name__ == "__main__":

    name_of_pc = "000016.pcd"
    cfg_file = 'cfgs/kitti_models/pv_rcnn.yaml'
    ckpt = "/app/checkpoints/pv_rcnn_8369.pth"

    up = UploadAPI(project_id=5268, ds_name="pointcloud")
    fpath = up.download_cloud(name_of_pc)

    res = predict(cfg_file, ckpt, fpath, '.pcd')
    bboxes, scores, labels = decode_prediction(res, thresh=0.8)

    annotation = Annotation.create_annotation(bboxes, labels)
    up.upload_annotation(annotation)
