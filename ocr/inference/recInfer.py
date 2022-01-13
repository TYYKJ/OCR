from __future__ import annotations

import cv2
import numpy as np
import torch

from ..rec import CRNN
from ..rec.utils import RecDataProcess, CTCLabelConverter

__all__ = ['RecInfer']


class RecInfer:
    def __init__(
            self,
            model_path: str,
            dict_path: str,
            std: float = 0.5,
            mean: float = 0.5,
            threshold: float = 0.7,
            device: str = 'cuda:0'
    ):
        # pl.seed_everything(1997)
        self.model_path = model_path
        self.dict_path = dict_path

        self.threshold = threshold

        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()
        # self.model.freeze()

        self.process = RecDataProcess(input_h=32, mean=mean, std=std)
        self.converter = CTCLabelConverter(dict_path)
        self.device = device

    def _load_model(self) -> CRNN:
        """
        加载模型
        :return:
        """
        return CRNN.load_from_checkpoint(self.model_path, alphabet_path=self.dict_path)

    def _predict(self, img: np.ndarray) -> list:
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        max_img_w = im.shape[1]
        im = self.process.normalize_img(im)
        im = self.process.resize_with_specific_height(im)
        im = self.process.width_pad_img(im, max_img_w)
        im = im.transpose([2, 0, 1])
        im = im[None]
        im = torch.from_numpy(im)

        im = im.to(self.device)
        with torch.no_grad():
            out = self.model(im)
            out = out.softmax(dim=2)
            out = out.cpu().numpy()
        result = [self.converter.decode(np.expand_dims(txt, 0)) for txt in out]
        return result

    def _filter_text(self, imgs: np.ndarray | list) -> list | None:
        result = self._predict(imgs)
        filter_result = []
        if len(result) != 0:
            for item in result:
                text, score = item[0]
                filter_result.append(''.join([x for index, x in enumerate(text) if score[index] > self.threshold]))
            return filter_result
        else:
            return

    def get_text(self, imgs: np.ndarray | list) -> list:
        return self._filter_text(imgs)
