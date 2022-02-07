import os

import cv2
from tqdm import tqdm


class RecDatasetTools:

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
                if i != len(cache) - 1:
                    f.write('\n')

    @staticmethod
    def split_dataset(txt_path):
        from sklearn.model_selection import train_test_split
        x, y = [], []
        with open(txt_path, 'r') as f:
            data = f.readlines()
            for item in data:
                x.append(item.split('\t')[0])
                y.append(item.split('\t')[1])
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1997)
        with open('train.txt', 'w') as f:
            for x, y in zip(X_train, y_train):
                y = y.strip('\n')
                txt = f'{x}\t{y}\n'
                f.write(txt)

        with open('val.txt', 'w') as f:
            for x, y in zip(X_test, y_test):
                y = y.strip('\n')
                txt = f'{x}\t{y}\n'
                f.write(txt)

    @staticmethod
    def check_rec_dataset(txt_path, img_path):
        annos = []
        with open(txt_path, 'r') as f:
            data = f.readlines()
            for item in tqdm(data):
                img_name, label = item.split('\t')
                im = cv2.imread(os.path.join(img_path, img_name))
                try:
                    _ = im.shape
                    # label = label.strip('\n')
                    # annos.append(f'{img_name}\t{label}\n')
                except AttributeError:
                    print(img_name)
        # with open('train.txt', 'w') as f:
        #     for item in annos:
        #         f.write(item)

    @staticmethod
    def del_info2(txt_path):
        with open(txt_path, 'r') as f:
            content = f.readlines()

        with open('train.txt', 'w') as f:
            for line in tqdm(content, desc='write info:'):
                data = line.split('\t')
                image_name = '/'.join(data[0].split('\\')[3:])
                label = data[1]
                write_content = image_name + '\t' + label + '\n'
                f.write(write_content)

    @staticmethod
    def get_img_mean_std(txt_path):
        with open(txt_path, 'r') as f:
            content = f.readlines()

        mean_list, std_list = [], []
        for line in tqdm(content, desc='write info:'):
            data = line.split('\t')
            image_name = '/media/cat/D/yellowPai0110/yellowPai/train/' + data[0]
            im = cv2.imread(image_name)

            try:
                (mean, stddv) = cv2.meanStdDev(im)
                mean_list.append(mean[0])
                std_list.append(stddv[0])
            except cv2.error:
                print(image_name)

        print(sum(mean_list) / len(mean_list) / 255)
        print(sum(std_list) / len(std_list) / 255)


if __name__ == '__main__':
    tool = RecDatasetTools()
    tool.make_dict('/home/cat/Documents/icdar2015-ok/recognition/train-no-space.txt')
    # tool.check_rec_dataset('/home/cat/Documents/all/train-no-space.txt', '/home/cat/Documents/all')
    # tool.del_info2('/home/cat/Documents/all/train.txt')
    # tool.check_rec_dataset('/home/cat/PycharmProjects/OCR/makeDataset/train.txt',
    #                        '/media/cat/D/CCPD2019/ccpd_base')
    # tool.get_img_mean_std('/media/cat/D/yellowPai0110/yellowPai/train.txt')
    # tool.get_img_mean_std('/home/cat/Documents/all/train.txt')
    # tool.split_dataset('/home/cat/PycharmProjects/OCR/makeDataset/ccpdData.txt')
    # tool.del_space('/home/cat/Documents/all/train.txt')
    # tool.del_info('/home/cat/Documents/icdar2017rctw/icdar2017/recognition/train.txt')
