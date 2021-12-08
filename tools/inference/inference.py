from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from classifyInfer import ClassifyInfer
from detInfer import DetInfer
from recInfer import RecInfer


class Inference:
    def __init__(
            self,
            det_model_path: str,
            rec_model_path: str,
            device: str,
            dict_path: str,
            classes: list | None,
            angle_model_path: str | None = None,
            std: float = 0.5,
            mean: float = 0.5,
            threshold: float = 0.7,
    ):
        self.det = DetInfer(det_model_path=det_model_path, device=device, threshold=threshold)
        self.rec = RecInfer(model_path=rec_model_path, dict_path=dict_path, batch_size=1,
                            std=std, mean=mean, threshold=threshold, device=device)
        self.angle = ClassifyInfer(classify_model_path=angle_model_path,
                                   class_names=classes) if angle_model_path else None

    def infer(self, img: Image | np.ndarray, img_save_name: str, cut_image_save_path: str | None = None) -> list:

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

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
