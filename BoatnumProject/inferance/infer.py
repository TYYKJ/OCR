# @Time    : 2021/8/6 下午2:27
# @Author  : cattree
# @File    : infer.py
# @Software: PyCharm
# @explain :
import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional

from ocr.CRNN import CRNNModel
from ocr.utils.CTCLabelConvert import CTCLabelConverter
from ocr.utils.convert import load

model = CRNNModel.CRNN(classes=5821,
                       character_path='/home/cattree/PycharmProjects/limapOCR/BoatnumProject/trainModel/dict.txt')

model = model.load_from_checkpoint('/home/cattree/PycharmProjects/limapOCR/BoatnumProject/trainModel/weights/limapOCR-BoatnumProject_trainModel/3ahqmpj5/checkpoints/epoch=1-step=4999.ckpt')
if torch.cuda.is_available():
    model.cuda()
data_list = load('/home/cattree/桌面/临时数据/20210802/360万中文数据集/360label/360_test.txt')[:10]
img_dir = '/home/cattree/桌面/临时数据/20210802/360万中文数据集/images'
convert = CTCLabelConverter(character='/home/cattree/PycharmProjects/limapOCR/BoatnumProject/trainModel/dict.txt')

txts = []
for item in data_list:

    line = item.split(' ')
    if img_dir.endswith('/'):
        img_path = img_dir + line[0]
    else:
        img_path = img_dir + f'/{line[0]}'
    image = torchvision.transforms.functional.to_tensor(PIL.Image.open(img_path))
    image = torch.unsqueeze(image, 0)
    label = line[1]

    device = 'cuda'
    with torch.no_grad():
        image = image.to(device)
        predict = model(image)
        predict = predict.softmax(dim=2)
    predict = predict.cpu().numpy()
    txts.extend([convert.decode(np.expand_dims(txt, 0)) for txt in predict])
    print(txts)
    print(label)
    txts = []

