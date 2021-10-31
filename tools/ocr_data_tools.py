# @Time    : 2021/9/29 下午12:42
# @Author  : 
# @File    : rec_data_tools
# @Software: PyCharm
# @explain :
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
    def move_file(source_path: str, obj_path: str):
        """
        移动文件

        :param source_path: 图片文件夹
        :param obj_path: 目标文件夹
        :return:
        """
        source_files = glob.glob(os.path.join(source_path, '*.jpg'))
        for file in tqdm(source_files):
            shutil.move(file, obj_path)

    @staticmethod
    def split_det_dataset(det_dataset_path: str):
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
            'data_root': '',
            'data_list': cache
        }

        train = {
            'data_root': '',
            'data_list': all_data
        }

        with open('val.json', 'w') as f:
            json.dump(val, f, indent=4)

        with open('val.json', 'w') as f:
            json.dump(train, f, indent=4)

    @staticmethod
    def check_img(json_path: str, img_path: str):
        with open(json_path, 'r') as f:
            data = f.read()

        data = json.loads(data)
        data_list = data['data_list']
        not_exist_img = []

        def get_shape(dl):
            for item in tqdm(dl, desc='img'):
                img = os.path.join(img_path, item['img_name'])

                im = cv2.imread(img)
                try:
                    _ = im.shape
                except AttributeError:
                    not_exist_img.append(item['img_name'])

        len_data_list = len(data_list) // 2
        d1 = data_list[:len_data_list]
        d2 = data_list[len_data_list:]
        t1 = threading.Thread(target=get_shape, args=(d1,))
        t2 = threading.Thread(target=get_shape, args=(d2,))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if not_exist_img:
            print(not_exist_img)
            inputs = input('will del img? (y or n)>')
            if inputs == 'y':
                for img_name in tqdm(not_exist_img, desc='del img'):
                    # if os.path.exists(os.path.join(img_path, item)):
                    #     os.remove(os.path.join(img_path, item))
                    for one_item in data_list:
                        if img_name == one_item['img_name']:
                            data_list.remove(one_item)
                with open('val.json', 'w') as f:
                    json.dump({'data_root': '', 'data_list': data_list}, f, indent=4)
        else:
            print('img is very gooooooooooooood!')

    @staticmethod
    def labelme_format_transform(parent_path):
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


class RecDatasetTools:

    @staticmethod
    def remove_space(txt_path):
        ...

    @staticmethod
    def del_info(txt_path):
        with open(txt_path, 'r') as f:
            content = f.readlines()

        with open('test.txt', 'w') as f:
            for line in tqdm(content, desc='write info:'):
                data = line.split('\t')
                image_name = data[0].split('\\')[-1]
                label = data[1]
                write_content = image_name + '\t' + label + '\n'
                f.write(write_content)

    @staticmethod
    def del_space(txt_path):
        """
        该函数仅支持 文件名\t标注 格式
        """
        with open(txt_path, 'r') as f:
            content = f.readlines()

        with open('train-no-space.txt', 'w') as f:
            for line in tqdm(content, desc='write info:'):
                data = line.split('\t')
                image_name = data[0]
                label = data[1].strip('\n')
                s = ''
                for char in label:
                    if char != ' ':
                        s += char
                write_info = image_name + '\t' + s + '\n'
                f.write(write_info)

    @staticmethod
    def make_dict(txt_path):
        with open(txt_path, 'r') as f:
            content = f.readlines()

        cache_dict = []
        for item in tqdm(content):
            line = item.split('\t')
            for i in line[-1].strip('\n'):
                if i not in cache_dict:
                    cache_dict.append(i)
        cache = list(set(cache_dict))
        cache = sorted(cache)
        print(cache)

        s = ''
        for item in cache:
            s += item
        print(s)
        print(len(s))

        with open(f'dict.txt', 'w') as f:
            for i in tqdm(range(len(cache))):
                if isinstance(cache[i], int):
                    d = str(cache[i])
                else:
                    d = cache[i]
                f.write(d)
                f.write('\n')


if __name__ == '__main__':
    jd = MakeDetJsonDataset()
    # jd.txt_to_json('/home/data/OCRData/det0902/val/gt', '/home/data/OCRData/MTWI2018/detection/val.json')
    # jd.labelme_format_transform('/home/cat/文档/渔船数据/Boatdet')
    # jd.split_det_dataset('/media/cat/data/OCRData/MTWI2018/detection/val.json')
    # jd.check_img('/home/cat/PycharmProjects/torch-ocr/tools/val.json', '/media/cat/data/OCRData/MTWI2018/detection/imgs')
    recd = RecDatasetTools()
    # recd.del_info('/home/cat/文档/icdar2015/recognition/test.txt')
    # recd.del_space('/home/cat/PycharmProjects/torch-ocr/tools/test.txt')
    recd.make_dict('/home/cat/文档/icdar2017/recognition/train-no-space.txt')
