import cv2
import yaml

from ocr import Inference

config = open('config.yaml', mode='r', encoding='utf-8')
config = yaml.load(config, Loader=yaml.FullLoader)

model = Inference(
    det_model_path=config['det_model_path'],
    rec_model_path=config['rec_model_path'],
    device=config['device'],
    dict_path=config['dict_path'],
    rec_std=0.5, rec_mean=0.5, threshold=0.7,
    angle_classes=config['angle_classes'],
    angle_classify_model_path=config['angle_model_path'],
    object_classes=None,
    object_classify_model_path=None
)
im = cv2.imread('/home/cat/Documents/ZF/XW/wsb20211019154616.jpg')
result = model.infer(
    img=im,
    img_save_name='1.jpg',
    cut_image_save_path=config['cut_image_save_path'],
    need_angle=config['need_angle'],
)
print(result)
