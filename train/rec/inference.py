# @Time    : 2021/9/18 上午11:17
# @Author  : 
# @File    : inference
# @Software: PyCharm
# @explain :

import os.path

import numpy as np
import torch

from BoatNumber.ocr_rec.train.dataloader.dataloader import RecDataProcess
from rec import CRNN
from rec.utils import CTCLabelConverter


class RecInfer:
    def __init__(self, model_path, batch_size=1):
        self.model = CRNN.load_from_checkpoint(model_path)
        self.model.eval()

        self.process = RecDataProcess(input_h=32, mean=0.5, std=0.5)
        self.converter = CTCLabelConverter('/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/dict.txt')
        self.batch_size = batch_size

    def predict(self, imgs):
        # 预处理根据训练来
        if not isinstance(imgs,list):
            imgs = [imgs]
        imgs = [self.process.normalize_img(self.process.resize_with_specific_height(img)) for img in imgs]
        widths = np.array([img.shape[1] for img in imgs])
        idxs = np.argsort(widths)
        txts = []
        for idx in range(0, len(imgs), self.batch_size):
            batch_idxs = idxs[idx:min(len(imgs), idx+self.batch_size)]
            batch_imgs = [self.process.width_pad_img(imgs[idx], imgs[batch_idxs[-1]].shape[1]) for idx in batch_idxs]
            batch_imgs = np.stack(batch_imgs)
            tensor = torch.from_numpy(batch_imgs.transpose([0,3, 1, 2])).float()
            with torch.no_grad():
                out = self.model(tensor)
                out = out.softmax(dim=2)
            out = out.cpu().numpy()
            txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
        #按输入图像的顺序排序
        idxs = np.argsort(idxs)
        out_txts = [txts[idx] for idx in idxs]
        return out_txts


if __name__ == '__main__':
    import cv2
    from rec.utils import load
    from tqdm import tqdm
    # args = init_args()
    data = load('/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0916datarec/test.txt')
    model = RecInfer(
        '/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/weights/torchOCR-BoatNumber_ocr_rec_train/l9m1xv9f/checkpoints/epoch=42-step=270082.ckpt')
    total = len(data)
    correct = 0
    not_correct_im = []
    for item in tqdm(data):
        im_name, label = item.split(' ')
        im_path = os.path.join('/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0916datarec/test', im_name)

        img = cv2.imread(im_path)
        out = model.predict(img)

        if out[0][0][0] == label:
            correct += 1
        else:
            not_correct_im.append(
                [im_name, label, out[0][0][0]]
            )
    for item in not_correct_im:
        print(item)
    print((correct / total) * 100, '%')
