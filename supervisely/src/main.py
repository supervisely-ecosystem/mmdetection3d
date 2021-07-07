from dotenv import dotenv_values

from mmdet3d.apis import inference_detector, init_model

config = {
    **dotenv_values(".env"),  # load shared development variables
    **dotenv_values(".env.secret")  # load sensitive variables
}

import os

for k, v in config.items():
    os.environ[k] = v

import logging
import os
import mmcv
import numpy as np
import supervisely_lib as sly
from supervisely_lib.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

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
    def pred_to_sly_geometry(labels, reverse=False):
        geometry = []
        for l in labels:
            x, y, z, dx, dy, dz, heading = l
            position = Vector3d(float(x), float(y), float(z * 0.5))

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


def decode_prediction(result, labels, score_thr, valid_labels=[]):
    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
        pred_labels = result[0]['pts_bbox']['labels_3d'].numpy()
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()
        pred_labels = result[0]['labels_3d'].numpy()
    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]
        pred_labels = pred_labels[inds]
        pred_scores = pred_scores[inds]

    pred_bboxes = pred_bboxes[:, :7]  # x, y, z, x_size, y_size, z_size, yaw

    labels = [labels[x] for x in pred_labels]  # convert int labels to str
    if any(valid_labels):
        mask = [l in ok_labels for l in labels]  # filter classes that dataset meta doesnt contain
        labels = np.array(labels)[mask]
        pred_scores = pred_scores[mask]
        pred_bboxes = pred_bboxes[mask]

    return pred_bboxes, pred_scores, labels


if __name__ == "__main__":
    # SECOND SecFPN [car]
    # config = "/mmdetection3d/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py"
    # checkpoint = "/data/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth"

    # SECOND SecFPN [3 classes]
    config = "/mmdetection3d/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py"
    checkpoint = "/data/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth"

    # CenterPoint not implemented yet
    # config = "/mmdetection3d/configs/centerpoint/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py"
    # checkpoint = "/data/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth"

    # PointPillars SecFPN [car]
    # config = "/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py"
    # checkpoint = "/data/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614-77663cd6.pth"

    # PointPillars SecFPN [3 classes]
    # config = "/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py"
    # checkpoint = "/data/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth"

    ok_labels = ['Car', 'Cyclist', 'Pedestrian']
    name_of_pc = "000016.pcd"
    up = UploadAPI(project_id=5268, ds_name="pointcloud")
    fpath = up.download_cloud(name_of_pc)

    model = init_model(config, checkpoint)
    labels = mmcv.Config.fromfile(config)['class_names']

    result, data = inference_detector(model, fpath)

    bboxes, scores, labels = decode_prediction(result, labels, score_thr=0.5)

    annotation = Annotation.create_annotation(bboxes, labels)
    up.upload_annotation(annotation)
