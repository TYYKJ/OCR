# @Time    : 2021/9/30 下午1:33
# @Author  : 
# @File    : 自动切图
# @Software: PyCharm
# @explain :

from torchvision import transforms
from det import DBPostProcess, ResizeShortSize, DBDetModel
import numpy as np
import torch


class DetInfer:
    def __init__(self, model_path, device='cuda:0'):
        self.device = device
        self.model = DBDetModel.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.freeze()

        self.resize = ResizeShortSize(736, False)
        self.post_process = DBPostProcess()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img):
        # 预处理根据训练来
        # model.to(device)
        # data.to(device)
        # with torch.no_grad():
        #     out = model(data)
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


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
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
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


if __name__ == '__main__':
    import cv2
    import time

    img = cv2.imread('/home/cat/PycharmProjects/torch-ocr/tools/inference/1.jpg')[130:-130, :, :]
    model = DetInfer('../train/weights/DB-resnet18-epoch=21-hmean=0.41-recall=0.28-precision=0.76.ckpt')

    start = time.time()
    bl, sl = model.predict(img)
    print(sl)
    if len(bl) != 0:
        imgs = [get_rotate_crop_image(img, box) for box in bl]
        for index, im in enumerate(imgs):
            # if im.shape[0] > 50:
            cv2.imwrite(f'{index}.jpg', im)
    else:
        print('no image')
    end = time.time()
    print(f'{end - start}s')
