# @Time    : 2021/8/24 下午2:12
# @Author  : cattree
# @File    : infer_ts
# @Software: PyCharm
# @explain :
import PIL
import cv2
import torch
import torchvision.transforms.functional
from torch import jit

from BoatNumber.ocr_rec.inference.ctc_decoder import ctc_decode

CHARS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ开养海渔烟牟福莱蓬长鲁'
model = jit.load('/home/cattree/PycharmProjects/torch-rec/BoatNumber/ocr_rec/inference/model_rec.ts')

model.eval()


def get_ocr_result(chars: str, model, img_path):
    def get_str(paths, predict):
        for path, pred in zip(paths, predict):
            text = ''.join(pred)
            return text

    CHAR2LABEL = {char: i + 1 for i, char in enumerate(chars)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    model.eval()
    im = cv2.imread(img_path)
    im = cv2.resize(im, (140, 32), interpolation=cv2.INTER_LINEAR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = PIL.Image.fromarray(im)

    im = torchvision.transforms.functional.to_tensor(im)
    im = torch.unsqueeze(im, dim=0)
    y_hat = model(im)
    log_probs = y_hat.log_softmax(dim=2)

    preds = ctc_decode(log_probs, method='beam_search', beam_size=10,
                       label2char=LABEL2CHAR)
    text = get_str(im, preds)
    return text


img_path = '/home/cattree/PycharmProjects/torch-ocr/BoatNumber/ocr_rec/inference/1.jpg'
print(get_ocr_result(CHARS, model, img_path))
