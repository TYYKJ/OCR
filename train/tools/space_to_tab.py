# @Time    : 2021/9/8 下午5:30
# @Author  : 
# @File    : transfer_data
# @Software: PyCharm
# @explain :
import json
import os
import pathlib


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': load_txt, '.json': load_json, '.list': load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


base_path = '/home/cattree/PycharmProjects/torchOCR/data'
split_type = ' '

files = os.listdir(base_path)
for file in files:
    if file.endswith('txt'):
        data = load(os.path.join(base_path, file))

        with open(file, 'w') as f:
            for item in data:
                line = item.split(split_type)
                filename, label = line[0], line[1]
                f.write(f'{filename}\t{label}\n')

