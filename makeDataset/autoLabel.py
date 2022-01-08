import json

import cv2
import numpy as np
import torch
from labelme import utils
from torchvision import transforms

from ocr.det import DBDetModel
from ocr.det.detmodules import ResizeShortSize
from ocr.det.postprocess import DBPostProcess


class AutoLabel:
    def __init__(
            self,
            det_model_path: str,
            label_save_path: str,
            device: str = 'cuda:0',
            threshold: float = 0.7,
    ):
        self.device = device
        self.label_save_path = label_save_path
        self.det_model_path = det_model_path
        self.model = self.load_model()
        self.model.to(device)
        self.model.eval()
        self.model.freeze()
        self.threshold = threshold
        self.resize = ResizeShortSize(736, False)
        self.post_process = DBPostProcess()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classification(self, img: np.ndarray):
        ...

    def load_model(self):
        return DBDetModel.load_from_checkpoint(self.det_model_path)

    @staticmethod
    def get_rotate_crop_image(img: np.ndarray, points: np.ndarray):
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
        m = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            m, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def predict_img(self, img):
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

    def filter_img(self, img: np.ndarray):

        height, weight = img.shape[0], img.shape[1]
        all_object = []
        box_list, score_list = self.predict_img(img)
        if len(box_list) != 0:
            box_list = [box_list[i] for i, score in enumerate(score_list) if score > self.threshold]
            for item in box_list:
                item = item.tolist()
                # points.append(item)
                all_object.append({
                    "label": "###",
                    "points": item,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
        else:
            print('无文字内容')
        return weight, height, all_object

    def general_annotation(self, img: np.ndarray,
                           img_name: str):
        weight, height, all_object = self.filter_img(img)

        info = {
            "version": "4.5.6",
            "flags": {},
            'shapes': all_object,
            "imagePath": img_name,
            "imageData": utils.img_arr_to_b64(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).decode('utf-8'),
            "imageHeight": height,
            "imageWidth": weight
        }
        json_str = json.dumps(info, indent=4)
        json_name = img_name.strip('.jpg')
        with open(f'{self.label_save_path}/{json_name}.json', 'w') as f:
            f.write(json_str)


if __name__ == '__main__':
    a = AutoLabel(
        det_model_path='/weights/DB-dpn68-epoch=11-hmean=0.42-recall=0.33-precision=0.56.ckpt',
        device='cpu',
        label_save_path='/home/cat/PycharmProjects/OCR/tools/inference'
    )

    image = cv2.imread('/home/cat/PycharmProjects/OCR/tools/inference/test.jpg')
    a.general_annotation(image, 'test.jpg')
