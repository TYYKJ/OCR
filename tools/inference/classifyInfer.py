from __future__ import annotations
import numpy as np
import torch
from torchvision import transforms
from ocr.classification import ClassificationModel
from PIL import Image
import pytorch_lightning as pl


class ClassifyInfer:
    def __init__(
            self,
            classify_model_path: str,
            class_names: list,
            device: str = 'cuda:0',
    ):
        pl.seed_everything(1997)
        self.device = device
        self.classify_model_path = classify_model_path
        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()
        self.model.freeze()
        self.classes_names = class_names

    def _load_model(self) -> ClassificationModel:
        return ClassificationModel.load_from_checkpoint(self.classify_model_path)

    def get_classification_result(self, img: np.ndarray | Image) -> str:
        if isinstance(img, np.ndarray):
            image = Image.fromarray(img)
            image = image.convert("RGB")
        else:
            image = img.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            pre = self.model(image.to(self.device))
        pre = torch.nn.functional.softmax(pre[0], dim=0)
        pre = pre.cpu().numpy()
        return self.classes_names[np.argmax(pre)]


# if __name__ == '__main__':
#     import cv2
#     imgt = cv2.imread('/home/cat/PycharmProjects/OCR/tools/inference/1.jpg')
#     c = ClassifyInfer(classify_model_path='/home/cat/PycharmProjects/OCR/weights/Classify-resnet18-epoch=00-val_acc=0.89.ckpt', class_names=['0', '180'])
#     name = c.get_classification_result(imgt)
#     print(name)
