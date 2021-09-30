# @Time    : 2021/9/29 下午12:42
# @Author  : 
# @File    : rec_data_tools
# @Software: PyCharm
# @explain :
import json
import glob
import os
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
                    points = [[int(item[0]), int(item[1])], [int(item[2]), int(item[3])], [int(item[4]), int(item[5])], [int(item[6]), int(item[7])]]
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

        with open('train.json', 'w') as f:
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


if __name__ == '__main__':
    jd = MakeDetJsonDataset()
    jd.txt_to_json('/home/data/OCRData/det0902/val/gt', '/home/data/OCRData/MTWI2018/detection/train.json')

