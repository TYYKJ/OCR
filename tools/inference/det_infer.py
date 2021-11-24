from torchvision import transforms
from det import DBPostProcess, ResizeShortSize, DBDetModel


class DetInfer:
    def __init__(self, model_path):
        self.model = DBDetModel.load_from_checkpoint(model_path)
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
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        # tensor = tensor.to(self.device)
        out = self.model(tensor)
        box_list, score_list = self.post_process(out, data['shape'])
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list

# if __name__ == '__main__':
#     import cv2
#     from matplotlib import pyplot as plt
#     from det.utils import draw_ocr_box_txt, draw_bbox
#
#     img = cv2.imread('1.jpg')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     model = DetInfer('../train/weights/DB-epoch=17-hmean=0.62.ckpt')
#     box_list, score_list = model.predict(img)
#     img = draw_ocr_box_txt(img, box_list)
#     img = draw_bbox(img, box_list)
#     plt.imshow(img)
#     plt.show()
