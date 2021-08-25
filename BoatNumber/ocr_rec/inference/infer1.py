# @Time    : 2021/8/18 上午8:37
# @Author  : cattree
# @File    : infer1
# @Software: PyCharm
# @explain :
import PIL
import torch
import torchvision.transforms.functional
import torch.hub
import cv2
from BoatNumber.ocr_rec.inference.ctc_decoder import ctc_decode

CHARS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ开养海渔烟牟福莱蓬长鲁'
# CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
# LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
#
#
# def show_result(paths, preds):
#     # print('\n===== result =====')
#     for path, pred in zip(paths, preds):
#         text = ''.join(pred)
#         return text
#
# import cv2
# import random
#
#
# def plot_one_box(x, img, color=None, label=None, line_thickness=None):
#     tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
#
#
# det_img = cv2.imread('/home/cattree/PycharmProjects/torch-rec/BoatNumber/ocr_rec/inference/1.jpg')
# det_model = torch.hub.load('ultralytics/yolov5', 'custom',
#                            path='/home/cattree/PycharmProjects/torch-rec/BoatNumber/ocr_rec/inference/CPSB.pt',
#                            force_reload=False, device='0')
# det_model.conf = 0.2
# det_model.iou = 0.2
# results = det_model(det_img, size=640)
# det = results.xyxy[0].cpu().detach().numpy()
# for *xyxy, obj_id, cls in reversed(det):
#     print(xyxy)
#     plot_one_box(xyxy, det_img)


# cv2.imshow('', det_img)
# cv2.waitKey(9000)

model = torch.load('/home/cattree/PycharmProjects/torch-rec/BoatNumber/ocr_rec/inference/ocr_rec.pt')
img_path = '/home/cattree/PycharmProjects/torch-ocr/BoatNumber/ocr_rec/inference/1.jpg'

model.eval()


#
# im = torchvision.transforms.functional.to_tensor(PIL.Image.open(img_path))
# im = torch.unsqueeze(im, dim=0)
# y_hat = model(im)
#
# log_probs = y_hat.log_softmax(dim=2)
#
# preds = ctc_decode(log_probs, method='beam_search', beam_size=10,
#                    label2char=LABEL2CHAR)
# text = show_result(im, preds)
# print(text)
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


print(get_ocr_result(CHARS, model, img_path))
