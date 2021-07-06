
import glob
from pathlib import Path

import os
os.chdir('/tmp/OpenPCDet/tools')
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import open3d as o3d
import logging

logger = logging.getLogger()



class Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)

        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            pcd = o3d.io.read_point_cloud(str(self.sample_file_list[index]))
            points = np.asarray(pcd.points)
            points = np.hstack((points, np.zeros(points.shape[0]).reshape((-1, 1))))
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def predict(cfg_file, ckpt, data_path, ext):
    cfg_from_yaml_file(cfg_file, cfg)

    logging.info(f"Processing pointcloud: {data_path}")
    logger.info('-----------------Run inference OpenPCDet-------------------------')
    dataset = Dataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    results = []
    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            logger.info(f'Processing index: \t{idx + 1}')
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            results.append(pred_dicts)

    logger.info('Inference done.')
    return results


def decode_prediction(res, thresh):
    bboxes = res[0][0]['pred_boxes'].cpu().numpy()  # [x, y, z, dx, dy, dz, heading]
    scores = res[0][0]['pred_scores'].cpu().numpy()
    labels = res[0][0]['pred_labels'].cpu().numpy()

    # Threshold cut
    indx = np.argwhere(scores > thresh)
    bboxes = np.squeeze(bboxes[indx])
    scores = np.squeeze(scores[indx])
    labels = np.squeeze(labels[indx])

    labels = np.array([cfg["CLASS_NAMES"][int(x-1)] for x in labels])  # to names
    return bboxes, scores, labels





