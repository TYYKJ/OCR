from __future__ import annotations
import numpy as np
import torch

from ocr.rec import CRNN
from ocr.rec.utils import RecDataProcess, CTCLabelConverter


class RecInfer:
    def __init__(
            self,
            model_path: str,
            dict_path: str,
            batch_size: int = 1,
            std: float = 0.5,
            mean: float = 0.5,
            threshold: float = 0.7,
            device: str = 'cuda:0'
    ):
        self.model_path = model_path
        self.dict_path = dict_path

        self.threshold = threshold

        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()
        self.model.freeze()

        self.process = RecDataProcess(input_h=32, mean=mean, std=std)
        self.converter = CTCLabelConverter(dict_path)
        self.batch_size = batch_size
        self.device = device

    def _load_model(self) -> CRNN:
        """
        加载模型
        :return:
        """
        return CRNN.load_from_checkpoint(self.model_path, alphabet_path=self.dict_path)

    def _predict(self, imgs: np.ndarray | list) -> list:
        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [self.process.normalize_img(self.process.resize_with_specific_height(img)) for img in imgs]
        widths = np.array([img.shape[1] for img in imgs])
        idxs = np.argsort(widths)
        txts = []
        for idx in range(0, len(imgs), self.batch_size):
            batch_idxs = idxs[idx:min(len(imgs), idx + self.batch_size)]
            batch_imgs = [self.process.width_pad_img(imgs[idx], imgs[batch_idxs[-1]].shape[1]) for idx in batch_idxs]
            batch_imgs = np.stack(batch_imgs)
            tensor = torch.from_numpy(batch_imgs.transpose([0, 3, 1, 2])).float()
            with torch.no_grad():
                out = self.model(tensor.to(self.device))
                out = out.softmax(dim=2)
            out = out.cpu().numpy()
            txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
        # 按输入图像的顺序排序
        idxs = np.argsort(idxs)
        out_txts = [txts[idx] for idx in idxs]
        return out_txts

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
        result = self._filter_text(imgs)
        return result


# if __name__ == '__main__':
#     import cv2
#     im = cv2.imread('/home/cat/Documents/icdar2015-ok/recognition/test/img_40_1.jpg')
#     rec = RecInfer(model_path='../../weights/CRNN-resnet50vd-epoch=109-val_acc=0.40.ckpt', dict_path='/home/cat/Documents/icdar2017rctw/icdar2017/recognition/dict.txt')
#     print(rec.get_text(im))
