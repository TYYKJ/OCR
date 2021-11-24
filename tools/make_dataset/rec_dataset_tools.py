import json
import glob
import os
import threading
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil


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
