# @Time    : 2021/8/2 上午9:43
# @Author  : cattree
# @File    : 字典生成
# @Software: PyCharm
# @explain :
from tqdm import tqdm

from rec.utils import load

file_path = '/BoatNumber/ocr_rec/data/BoaData/plate_train/txt.txt'

data = load(file_path)
cache_dict = []

for item in tqdm(data):
    line = item.split(' ')
    for i in line[-1]:
        if i not in cache_dict:
            cache_dict.append(i)

cache = list(set(cache_dict))

cache = sorted(cache)

with open('dict.txt', 'w') as f:
    for i in tqdm(range(len(cache))):
        if isinstance(cache[i], int):
            d = str(cache[i])
        else:
            d = cache[i]
        f.write(d)
        f.write('\n')
