import os
from mmdet3d.apis import inference_detector, init_model
import numpy as np
import supervisely_lib as sly
from supervisely_lib.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
import sly_globals as g
import mmcv


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
    def create_annotation(bboxes, labels, meta):
        geometry_list = Annotation.pred_to_sly_geometry(bboxes)
        figures = []
        objs = []

        for label, geometry in zip(labels, geometry_list):  # by object in point cloud
            pcobj = sly.PointcloudObject(meta.get_obj_class(label))
            figures.append(sly.PointcloudFigure(pcobj, geometry))
            objs.append(pcobj)

        pc_annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(objs), figures)
        return pc_annotation


def _download_dir(remote_dir, local_dir):
    remote_files = g.api.file.list2(g.team_id, remote_dir)
    progress = sly.Progress(f"Downloading {remote_dir}", len(remote_files), need_info_log=True)
    for remote_file in remote_files:
        local_file = os.path.join(local_dir, sly.fs.get_file_name_with_ext(remote_file.path))
        if sly.fs.file_exists(local_file):  # @TODO: for debug
            pass
        else:
            g.api.file.download(g.team_id, remote_file.path, local_file)
        progress.iter_done_report()


@sly.timeit
def download_model_and_configs():
    # remote_model_dir, remote_model_weights_name = os.path.split(g.remote_weights_path)
    # remote_model_index = sly.fs.get_file_name(g.remote_weights_path) + '.index'
    # remote_config_dir = remote_model_dir
    # # Load config ../../../info/*.yml  (assert unique yml in dir)
    # for i in range(3):
    #     remote_config_dir = os.path.split(remote_config_dir)[0]
    # remote_config_dir = os.path.join(remote_config_dir, 'info')
    # info_file_list = g.api.file.list(g.team_id, remote_config_dir)
    # config = [x['name'] for x in info_file_list if x['name'].endswith('yml')]
    # assert len(config) == 1
    # remote_config_file = os.path.join(remote_config_dir, config[0])
    #
    # g.local_weights_path = os.path.join(g.my_app.data_dir, remote_model_weights_name)
    # g.local_index_path = os.path.join(g.my_app.data_dir, remote_model_index)
    # g.local_model_config_path = os.path.join(g.my_app.data_dir, config[0])
    #
    # g.api.file.download(g.team_id, g.remote_weights_path, g.local_weights_path)
    # g.api.file.download(g.team_id, os.path.join(remote_model_dir, remote_model_index), g.local_index_path)
    # g.api.file.download(g.team_id, remote_config_file, g.local_model_config_path)

    g.local_weights_path = "/data/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d_20201016_220844-3058d9fc.pth"
    g.local_model_config_path = "/mmdetection3d/configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py"


    #sly.logger.debug(f"Remote weights {g.remote_weights_path}")
    sly.logger.debug(f"Local weights {g.local_weights_path}")
    #sly.logger.debug(f"Local index {g.local_index_path}")
    sly.logger.debug(f"Local config path {g.local_model_config_path}")
    sly.logger.info("Model has been successfully downloaded")




def construct_model_meta():
    labels = mmcv.Config.fromfile(g.local_model_config_path)['class_names']

    g.gt_index_to_labels = dict(enumerate(labels))
    g.gt_labels = {v: k for k, v in g.gt_index_to_labels.items()}

    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection([sly.ObjClass(k, Cuboid3d) for k in labels]))
    sly.logger.info(g.meta.to_json())


def find_unique_file(dir_where, endswith):
    files = [x for x in os.listdir(dir_where) if x.endswith(endswith)]
    if not files:
        sly.logger.error(f'No {endswith} file found in {dir_where}!')
    elif len(files) > 1:
        sly.logger.error(f'More than one {endswith} file found in {dir_where}\n!')
    else:
        return os.path.join(dir_where, files[0])
    return None


@sly.timeit
def deploy_model():
    file = g.local_weights_path
    if os.path.exists(file):

        g.model = init_model(g.local_model_config_path, g.local_weights_path)
        sly.logger.info("Model has been successfully deployed")
    else:
        msg = f"Wrong model path: {file}!"
        sly.logger.error(msg)
        raise ValueError(msg)


def decode_prediction(result, labels, score_thr):
    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
        pred_labels = result[0]['pts_bbox']['labels_3d'].numpy()
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()
        pred_labels = result[0]['labels_3d'].numpy()

    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]
        pred_labels = pred_labels[inds]
        pred_scores = pred_scores[inds]

    pred_bboxes = pred_bboxes[:, :7]  # x, y, z, x_size, y_size, z_size, yaw
    labels = [labels[x] for x in pred_labels]  # convert int labels to str
    return pred_bboxes, pred_scores, labels


def inference_model(model, local_pointcloud_path, thresh=0.3):
    """Inference 1 pointcloud with the detector.

    Args:
        model (nn.Module): The loaded detector (ObjectDetection pipeline instance).
        local_pointcloud_path: str: The pointcloud filename.
    Returns:
        result Pointcloud.annotation object`.
    """
    result, data = inference_detector(model, local_pointcloud_path)
    pred_bboxes, pred_scores, labels = decode_prediction(result, g.gt_index_to_labels, thresh)
    annotation = Annotation.create_annotation(pred_bboxes, labels, g.meta)
    return annotation
