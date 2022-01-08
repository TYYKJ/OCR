from tqdm import tqdm

from .utils import *


def cvt(gt_path, save_path, img_folder):
    """
    将icdar2015格式的gt转换为json格式
    :param gt_path:
    :param save_path:
    :return:
    """
    gt_dict = {'data_root': img_folder}
    data_list = []
    origin_gt = load(gt_path)
    for img_name, gt in tqdm(origin_gt.items()):
        cur_gt = {'img_name': img_name + '.jpg', 'annotations': []}
        for line in gt:
            cur_line_gt = {'polygon': [], 'text': '', 'illegibility': False, 'language': 'Latin'}
            chars_gt = [{'polygon': [], 'char': '', 'illegibility': False, 'language': 'Latin'}]
            cur_line_gt['chars'] = chars_gt
            # 字符串级别的信息
            cur_line_gt['polygon'] = line['points']
            cur_line_gt['text'] = line['transcription']
            cur_line_gt['illegibility'] = line['illegibility']
            cur_gt['annotations'].append(cur_line_gt)
        data_list.append(cur_gt)
    gt_dict['data_list'] = data_list
    save(gt_dict, save_path)


if __name__ == '__main__':
    gt_path = '/home/cat/Downloads/train_full_labels.json'
    img_folder = '/home/cat/Downloads/image'
    save_path = '/home/cat/Downloads/train.json'
    cvt(gt_path, save_path, img_folder)
