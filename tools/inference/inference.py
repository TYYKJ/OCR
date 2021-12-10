from __future__ import annotations

import cv2
import numpy as np
import yaml
from PIL import Image

from tools.inference.classifyInfer import ClassifyInfer
from tools.inference.detInfer import DetInfer
from tools.inference.recInfer import RecInfer
from service_streamer import ManagedModel


class Inference:
    def __init__(
            self,
            det_model_path: str,
            rec_model_path: str,
            angle_model_path: str | None,
            device: str,
            dict_path: str,
            classify_classes: list | None,
            std: float = 0.5,
            mean: float = 0.5,
            threshold: float = 0.7,
    ):
        self.det = DetInfer(det_model_path=det_model_path, device=device, threshold=threshold)
        self.rec = RecInfer(model_path=rec_model_path, dict_path=dict_path, batch_size=1,
                            std=std, mean=mean, threshold=threshold, device=device)
        self.angle = ClassifyInfer(classify_model_path=angle_model_path,
                                   class_names=classify_classes) if angle_model_path else None

    def predict(self, batch: list) -> list:
        img = batch[0][0]
        cut_image_save_path = batch[0][1]
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        result = []

        cut_imgs = self.det.get_img_text_area(img)
        if cut_imgs:
            for index, cut_img in enumerate(cut_imgs):
                if self.angle:
                    if self.angle.get_classification_result(cut_img) == '0':
                        if cut_image_save_path:
                            cv2.imwrite(f'{cut_image_save_path}/{index}.jpg', cut_img)
                        result.append(self.rec.get_text(cut_img))
                    else:
                        cut_img = cv2.flip(cut_img, -1)
                        if cut_image_save_path:
                            cv2.imwrite(f'{cut_image_save_path}/{index}.jpg', cut_img)
                        result.append(self.rec.get_text(cut_img))
                else:
                    if cut_image_save_path:
                        cv2.imwrite(f'{cut_image_save_path}/{index}.jpg', cut_img)
                    result.append(self.rec.get_text(cut_img))

        return result


class ManagedBertModel(ManagedModel):

    def init_model(self):
        stream = open('flask.yaml', mode='r', encoding='utf-8')
        data = yaml.load(stream, Loader=yaml.FullLoader)

        self.model = Inference(det_model_path=data['det_model_path'],
                               rec_model_path=data['rec_model_path'],
                               angle_model_path=data['angle_model_path'], device=data['device'],
                               dict_path=data['dict_path'],
                               classify_classes=data['classes'], std=0.5, mean=0.5, threshold=0.7)

    def predict(self, batch):
        return self.model.predict(batch)
