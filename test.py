import os
import shutil

# base = '/home/data/OCRData/icdar2017rctw/detection/imgs'
# label_path = '/home/data/OCRData/icdar2017rctw/gt'
# im_path = '/home/data/OCRData/icdar2017rctw/image'
#
# files = os.listdir(base)
#
# labels = []
# imgs = []
#
#
# for item in tqdm(files):
#     if item.endswith('txt'):
#         shutil.move(f'{base}/{item}', f'{label_path}/{item}')
#     elif item.endswith('jpg'):
#         shutil.move(f'{base}/{item}', f'{im_path}/{item}')

train_gt_path = '/home/data/OCRData/icdar2017rctw/detection/train/gt'
train_image_path = '/home/data/OCRData/icdar2017rctw/detection/train/image'

val_gt_path = '/home/data/OCRData/icdar2017rctw/detection/val/gt'
val_image_path = '/home/data/OCRData/icdar2017rctw/detection/val/image'

images = []
files = os.listdir(train_image_path)
for item in files:
    if item.endswith('jpg'):
        images.append(item)

val_num = int(len(images) * 0.2)

#
for i in range(val_num):
    # print(files[i][:-3])
    shutil.move(f'{train_gt_path}/{files[i][:-3]}txt', val_gt_path)
    shutil.move(f'{train_image_path}/{files[i]}', val_image_path)
