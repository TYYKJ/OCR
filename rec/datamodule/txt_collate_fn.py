# @Time    : 2021/9/26 下午3:32
# @Author  : 
# @File    : txt_collate_fn
# @Software: PyCharm
# @explain :
import numpy as np
import torch
from torchvision import transforms


class RecCollateFn:
    def __init__(self, process_callback):
        self.process = process_callback
        self.t = transforms.ToTensor()

    def __call__(self, batch):
        resize_images = []

        all_same_height_images = [self.process.resize_with_specific_height(_['img']) for _ in batch]
        max_img_w = max({m_img.shape[1] for m_img in all_same_height_images})
        # make sure max_img_w is integral multiple of 8
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        labels = []
        for i in range(len(batch)):
            _label = batch[i]['label']
            labels.append(_label)
            img = self.process.normalize_img(all_same_height_images[i])
            img = self.process.width_pad_img(img, max_img_w)

            img = img.transpose([2, 0, 1])
            resize_images.append(torch.tensor(img, dtype=torch.float))
        resize_images = torch.stack(resize_images)
        return {'img': resize_images, 'label': labels}
