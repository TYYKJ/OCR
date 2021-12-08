from __future__ import annotations

from typing import Tuple, List, Any

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torchvision import transforms

from ocr import DBDetModel
from ocr.det.detmodules import ResizeShortSize
from ocr.det.postprocess import DBPostProcess

__all__ = ['DetInfer']


class DetInfer:
    def __init__(
            self,
            det_model_path: str,
            device: str = 'cuda:0',
            threshold: float = 0.7
    ):
        pl.seed_everything(1997)
        self.device = device
        self.det_model_path = det_model_path
        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()
        self.model.freeze()
        self.threshold = threshold
        self.resize = ResizeShortSize(736, False)
        self.post_process = DBPostProcess()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self) -> DBDetModel:
        """
        加载模型
        :return: DBDetModel类
        """
        return DBDetModel.load_from_checkpoint(self.det_model_path)

    @staticmethod
    def _get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        旋转

        :param img:
        :param points:
        :return:
        """
        points = points.astype(np.float32)
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        m = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            m, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def _predict(self, img: np.ndarray) -> Tuple[List[Any], List[Any]]:
        """
        预测图像
        :param img: 图像
        :return: 元组
        """
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        box_list, score_list = self.post_process(out.cpu(), data['shape'])
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list

    def _filter_img(self, img: np.ndarray) -> list | None:
        """
        过滤低于阈值的文本图像

        :param img: 图像
        :return: 高于阈值的图像列表或者None
        """

        box_list, score_list = self._predict(img)
        if len(box_list) != 0:
            box_list = [box_list[i] for i, score in enumerate(score_list) if score > self.threshold]
            return box_list
        else:
            return

    def get_img_text_area(self, img: np.ndarray) -> list:
        """
        获取文本区域图像

        :param img: 检测图像
        :return: 文本图像列表
        """
        box_list = self._filter_img(img)

        return [self._get_rotate_crop_image(img, box) for box in box_list] if box_list else None
