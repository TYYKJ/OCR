# @Time    : 2021/9/18 上午11:17
# @Author  :
# @File    : inference
# @Software: PyCharm
# @explain :
import os

import numpy as np
import torch
from rec import CRNN, RecDataProcess, CTCLabelConverter


class RecInfer:
    def __init__(self, model_path, dict_path, batch_size=1):
        CRNN(charset_path=dict_path, classes=18)
        self.model = CRNN.load_from_checkpoint(model_path, charset_path=dict_path)
        self.model.eval()
        self.model.freeze()

        self.process = RecDataProcess(input_h=32, mean=0.5, std=0.5)
        self.converter = CTCLabelConverter(dict_path)
        self.batch_size = batch_size

    def predict(self, imgs):
        # 预处理根据训练来
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
                out = self.model(tensor)
                out = out.softmax(dim=2)
            out = out.cpu().numpy()
            txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
        # 按输入图像的顺序排序
        idxs = np.argsort(idxs)
        out_txts = [txts[idx] for idx in idxs]
        return out_txts


if __name__ == '__main__':
    import cv2
    from rec.utils import load
    from tqdm import tqdm

    # args = init_args()
    data = load('/home/cat/文档/newcut/valcut.txt')
    model = RecInfer(
        '/home/cat/PycharmProjects/torch-ocr/tools/train/weights/CRNN-epoch=31-val_loss=2.82--eval_acc=0.31.ckpt',
        dict_path='/home/cat/PycharmProjects/torch-ocr/tools/train/rec_train/dict.txt')
    total = len(data)
    correct = 0
    not_correct_im = []
    for item in tqdm(data[:100]):
        im_name, label = item.split('\t')
        im_path = os.path.join('/home/cat/文档/newcut/valcut',
                               im_name)

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
