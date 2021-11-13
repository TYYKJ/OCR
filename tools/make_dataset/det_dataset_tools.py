import json
import glob
import os
import threading
import cv2
from tqdm import tqdm
import shutil


class MakeDetJsonDataset:

    @staticmethod
    def txt_to_json(txt_path: str, json_path: str):
        """
        将数据加入json文件中

        :param txt_path: txt文件夹路径
        :param json_path: 需要加入的json文件路径
        :return:
        """
        assert os.path.isfile(json_path), 'Json file is not exist!'
        with open(json_path, 'r') as jf:
            json_data = jf.read()

        json_data = json.loads(json_data)

        all_txt_files = glob.glob(os.path.join(txt_path, '*.txt'))
        for file in tqdm(all_txt_files):
            filename = file.split('/')[-1]
            with open(file, 'r') as f:
                txt_data = f.readlines()
                for item in txt_data:
                    item = item.strip('\n').split(',')
                    points = [[int(item[0]), int(item[1])], [int(item[2]), int(item[3])], [int(item[4]), int(item[5])],
                              [int(item[6]), int(item[7])]]
                    label = item[-1]

                    if int(item[-2]):
                        illegibility = True
                    else:
                        illegibility = False

                    one_obj = json.dumps({
                        'img_name': filename,
                        'annotations': points,
                        'text': label,
                        'illegibility': illegibility,
                        "language": "Latin",
                        "chars": [
                            {
                                "polygon": [],
                                "char": "",
                                "illegibility": illegibility,
                                "language": "Latin"
                            }
                        ]
                    })
                    json_data['data_list'].append(one_obj)

        with open('val.json', 'w') as f:
            json.dump(json_data, f)

    @staticmethod
    def move_img_file(source_paths: list, obj_path: str):
        """
        移动文件

        :param source_paths: 图片文件夹
        :param obj_path: 目标文件夹
        :return:
        """

        for source_path in source_paths:
            source_files = os.listdir(source_path)
            if source_files:
                for file in tqdm(source_files):
                    if not file.endswith('txt'):
                        shutil.copy(os.path.join(source_path, file), obj_path)
            else:
                print(source_path + '没有图片')

    @staticmethod
    def concat_ocr_json(json_file_path_list: list, img_path, json_save_path):
        data_list = []
        for label_file in tqdm(json_file_path_list, desc='合并json'):
            with open(label_file, 'r') as f:
                data = json.loads(f.read()).get('data_list')
                for item in data:
                    data_list.append(item)
        cache = {
            'data_root': img_path,
            'data_list': data_list
        }
        with open(json_save_path, 'w') as f:
            json.dump(cache, f, indent=4)

    @staticmethod
    def remove_img(img_path):
        os.remove(img_path)

    def concat_ocr_detect_dataset(self,
                                  img_source_paths: list,
                                  img_obj_path: str,
                                  concat_json_save_path: str,
                                  json_file_paths: list):
        self.move_img_file(img_source_paths, img_obj_path)
        self.concat_ocr_json(json_file_paths, img_obj_path, concat_json_save_path)

    @staticmethod
    def split_det_dataset(det_dataset_path: str, img_path):
        import random
        with open(det_dataset_path, 'r') as f:
            data = f.read()

        data = json.loads(data)
        all_data = data['data_list']
        cache = []
        for _ in tqdm(range(int(len(all_data) * 0.2))):
            cache_data = random.choice(all_data)
            all_data.remove(cache_data)
            cache.append(cache_data)

        val = {
            'data_root': f'{img_path}',
            'data_list': cache
        }

        train = {
            'data_root': f'{img_path}',
            'data_list': all_data
        }

        with open('val.json', 'w') as f:
            json.dump(val, f, indent=4)

        with open('train.json', 'w') as f:
            json.dump(train, f, indent=4)

    @staticmethod
    def check_img(json_path: str, img_path: str, save_path: str):
        with open(json_path, 'r') as f:
            data = f.read()

        data = json.loads(data)
        data_list = data['data_list']
        not_exist_img = []

        def get_shape(dl, count):
            for item in tqdm(dl, desc=f'img process-{count}'):
                img = os.path.join(img_path, item['img_name'])

                im = cv2.imread(img)
                try:
                    _ = im.shape
                except AttributeError:
                    not_exist_img.append(item['img_name'])

        len_data_list = len(data_list) // 2
        d1 = data_list[:len_data_list]
        d2 = data_list[len_data_list:]

        # d3 = d1[:len(d1) // 2]
        # d4 = d1[len(d1) // 2:]
        #
        # d5 = d2[:len(d2) // 2]
        # d6 = d2[len(d2) // 2:]

        t1 = threading.Thread(target=get_shape, args=(d1, '1'))
        t2 = threading.Thread(target=get_shape, args=(d2, '2'))
        # t3 = threading.Thread(target=get_shape, args=(d5,'3'))
        # t4 = threading.Thread(target=get_shape, args=(d6,'4'))

        t1.start()
        t2.start()
        # t3.start()
        # t4.start()
        t1.join()
        t2.join()
        # t3.join()
        # t4.join()
        print(f'总数量:{len(data_list)}')
        if not_exist_img:
            print(len(not_exist_img))
            inputs = input('will del img? (y or n)>')
            if inputs == 'y':
                for img_name in tqdm(not_exist_img, desc='del img'):
                    # if os.path.exists(os.path.join(img_path, item)):
                    #     os.remove(os.path.join(img_path, item))
                    for one_item in data_list:
                        if img_name == one_item['img_name']:
                            data_list.remove(one_item)
                with open(save_path, 'w') as f:
                    json.dump({'data_root': img_path, 'data_list': data_list}, f, indent=4)
        else:
            print('img is very gooooooooooooood!')

    @staticmethod
    def labelme_format_transform(parent_path):
        """
        转换labelme格式到ocr格式
        """
        cache = {
            'data_root': '',
            'data_list': []
        }
        files = glob.glob(os.path.join(parent_path, '*.json'))
        for file in files:
            with open(file, 'r') as f:
                data = json.loads(f.read())
                img_name = data['imagePath']
                annotations = []
                for item in data['shapes']:
                    annotations.append(
                        {
                            'polygon': item['points'],
                            'text': item['label'],
                            'illegibility': False,
                            "language": "Latin",
                            "chars": [
                                {
                                    "polygon": [],
                                    "char": "",
                                    "illegibility": False,
                                    "language": "Latin"
                                }
                            ]
                        }
                    )
                cache['data_list'].append(
                    {
                        'img_name': img_name,
                        'annotations': annotations
                    }
                )

        with open('train.json', 'w') as f:
            json.dump(cache, f, indent=4)


if __name__ == '__main__':
    det = MakeDetJsonDataset()

    # det.split_det_dataset(det_dataset_path='/home/cat/Documents/PreTrainOCRData/train.json',
    #                       img_path='/home/cat/Documents/PreTrainOCRData/image')
    # det.check_img(
    #     '/home/cat/Documents/PreTrainOCRData/train.json',
    #     img_path='/home/cat/Documents/PreTrainOCRData/image',
    #     save_path='/home/cat/Documents/PreTrainOCRData/train.json'
    # )
    det.check_img(
        '/home/cat/Documents/PreTrainOCRData/val.json',
        img_path='/home/cat/Documents/PreTrainOCRData/val_img',
        save_path='/home/cat/Documents/PreTrainOCRData/val.json'
    )

    # det.concat_ocr_detect_dataset(
    #     img_source_paths=[
    #         '/home/cat/Documents/COCO_Text-ok/detection/val',
    #         '/home/cat/Documents/icdar2015-ok/detection/test/imgs',
    #     ],
    #     img_obj_path='/home/cat/Documents/PreTrainOCRData/val_img',
    #     json_file_paths=[
    #         '/home/cat/Documents/COCO_Text-ok/detection/val.json',
    #         '/home/cat/Documents/icdar2015-ok/detection/test.json',
    #     ],
    #     concat_json_save_path='/home/cat/Documents/PreTrainOCRData/val.json'
    # )

    # det.concat_ocr_detect_dataset(
    #     img_source_paths=[
    #         '/home/cat/Documents/Art/detection/train_images',
    #         '/home/cat/Documents/COCO_Text-ok/detection/train',
    #         '/home/cat/Documents/icdar2015-ok/detection/train/imgs',
    #         '/home/cat/Documents/icdar2017rctw/icdar2017/detection/imgs',
    #         '/home/cat/Documents/LSVT/lstvall/detection/imgs',
    #         '/home/cat/Documents/ReCTS/detection/img',
    #     ],
    #     img_obj_path='/home/cat/Documents/PreTrainOCRData/image',
    #     json_file_paths=[
    #         '/home/cat/Documents/Art/detection/train.json',
    #         '/home/cat/Documents/COCO_Text-ok/detection/train.json',
    #         '/home/cat/Documents/icdar2015-ok/detection/train.json',
    #         '/home/cat/Documents/icdar2017rctw/icdar2017/detection/train.json',
    #         '/home/cat/Documents/ReCTS/detection/train.json',
    #     ],
    #     concat_json_save_path='/home/cat/Documents/PreTrainOCRData/train.json'
    # )
