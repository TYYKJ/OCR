from tqdm import tqdm

from rec.utils import load

file_path = '/home/cattree/PycharmProjects/torchOCR/BoatNumber/ocr_rec/data/0909recDatanoresize/recDatanoresize/trainnoresize.txt'

data = load(file_path)
cache_dict = []

for item in tqdm(data):
    line = item.split(' ')
    for i in line[-1]:

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

with open('dict.txt', 'w') as f:
    for i in tqdm(range(len(cache))):
        if isinstance(cache[i], int):
            d = str(cache[i])
        else:
            d = cache[i]
        f.write(d)
        f.write('\n')
