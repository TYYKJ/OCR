from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from .classifyInfer import ClassifyInfer
from .detInfer import DetInfer
from .recInfer import RecInfer


class Inference:
    """OCR两阶段推理

    Attributes:
        det_model_path: 检测模型路径
        rec_model_path: 识别模型路径
        device：推理设备
        dict_path：字典路径
        classify_classes：分类类别数
        angle_model_path：角度分类模型路径
        std：方差
        mean：均值
        threshold：阈值
    """

    def __init__(
            self,
            det_model_path: str,
            rec_model_path: str,
            device: str,
            dict_path: str,
            classify_classes: str | None = None,
            angle_model_path: str | None = None,
            std: float = 0.5,
            mean: float = 0.5,
            threshold: float = 0.7,
    ):
        if classify_classes is None:
            classify_classes = []
        self.det = DetInfer(det_model_path=det_model_path, device=device, threshold=threshold)
        self.rec = RecInfer(model_path=rec_model_path, dict_path=dict_path, batch_size=1,
                            std=std, mean=mean, threshold=threshold, device=device)
        self.angle = ClassifyInfer(classify_model_path=angle_model_path,
                                   class_names=classify_classes) if angle_model_path else None

    def infer(self, img: Image | np.ndarray, img_save_name: str, cut_image_save_path: str | None = None) -> list:
        """
        返回推理结果

        Args:
            img: 图像
            img_save_name: 保存名称
            cut_image_save_path: 保存路径

        Returns:
            返回文字列表或空列表
        """
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        result = []

        cut_imgs = self.det.get_img_text_area(img)
        if cut_imgs:
            for index, cut_img in enumerate(cut_imgs):
                if self.angle:
                    if self.angle.get_classification_result(cut_img) == '0':
                        if cut_image_save_path:
                            cv2.imwrite(os.path.join(cut_image_save_path, f'{img_save_name}_{index}.jpg'), cut_img)
                        result.append(self.rec.get_text(cut_img))
                    else:
                        cut_img = cv2.flip(cut_img, -1)
                        if cut_image_save_path:
                            cv2.imwrite(os.path.join(cut_image_save_path, f'{img_save_name}_{index}.jpg'), cut_img)
                        result.append(self.rec.get_text(cut_img))
                else:
                    if cut_image_save_path:
                        cv2.imwrite(os.path.join(cut_image_save_path, f'{img_save_name}_{index}.jpg'), cut_img)
                    result.append(self.rec.get_text(cut_img))

        return result
