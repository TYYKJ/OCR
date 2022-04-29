from ocr import RecInfer
import cv2
import os
import pandas as pd
from tqdm import tqdm

infer = RecInfer(
    model_path='../weights/hs512CycleLR-CRNN-resnet101vd-epoch=64-val_acc=0.90.ckpt',
    device='cuda:1',
    dict_path='/home/cat/PycharmProjects/OCR/makeDataset/dict.txt',
    threshold=0.2
)


all_result = []
num = []

error_num = 0
len_exceed_four = 0
for item in tqdm(range(1, 25001)):
    path = os.path.join('/home/cat/Documents/验证码/test_dataset', str(item)+'.png')
    im = cv2.imread(path)
    result = infer.get_text(im)

    if len(result[0]) < 4:
        error_num += 1
    if len(result[0]) > 4:
        result = result[0][:-1]
        error_num += 1
        len_exceed_four += 1
    all_result.append(result[0])
    num.append(item)

print(error_num)
print(len_exceed_four)

df = pd.DataFrame({
    'num': num,
    'tag': all_result
})

df.to_csv('result.csv', index=False)
