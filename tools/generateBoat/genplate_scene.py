# coding=utf-8
"""
生成船牌数据，将船牌放到自然图像中
"""
import os
import numpy as np
import cv2
import argparse
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from math import *
from PlateCommon import *

TEMPLATE_IMAGE = "./images/1.bmp"


def gen_plate_string():
    """生成船牌号码字符串"""
    plate_str = ""
    for cpos in range(8):
        if cpos == 0:
            plate_str += chars[15]
        elif cpos == 1:
            plate_str += chars[65 + r(6)]
        elif cpos == 2:
            plate_str += chars[71]
        else:
            plate_str += chars[31 + r(10)]
    return plate_str


class GenPlateScene:

    """船牌数据生成器，船牌放在自然场景中，位置信息存储在txt文件中"""
    def __init__(self, fontCh, fontEng, NoPlates):
        """
        :param fontCh: 中文字体
        :param fontEng: 英文字体
        :param NoPlates: 场景图片文件夹
        """
        self.fontC = ImageFont.truetype(fontCh, 43, 0)
        self.fontE = ImageFont.truetype(fontEng, 28, 0)
        self.img = np.array(Image.new("RGB", (130, 32), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread(TEMPLATE_IMAGE), (130, 32))
        self.noplates_path = []
        for parent, _, filenames in os.walk(NoPlates):
            for filename in filenames:
                self.noplates_path.append(parent + "/" + filename)

    def draw(self, val):
        """
        设置车牌文字位置
        :param val:船牌号
        :return:返回合成之后的船牌图片
        """
        offset = 0
        self.img[0:32, offset + 8:offset + 8 + 18] = GenCh(self.fontC, val[0])
        self.img[0:32, offset + 8 + 18:offset + 8 + 18 + 18] = GenCh(self.fontC, val[1])
        self.img[0:32, offset + 8 + 18 + 18:offset + 8 + 18 + 18 + 18] = GenCh(self.fontC, val[2])
        for i in range(5):
            base = offset + 8 + 18 + 18 + 18 + i * 12
            self.img[0:32, base:base + 12] = GenCh1(self.fontE, val[i + 3])
        return self.img

    def generate(self, text):
        """
        对船牌号文字和图片进行旋转变形等操作
        :param text:生成的船牌号
        :return:图像和位置信息
        """
        print(text, len(text))
        fg = self.draw(text.encode(encoding="utf-8").decode(encoding="utf-8"))  # 得到白底黑字
        fg = cv2.bitwise_not(fg)  # 得到黑底白字
        com = cv2.bitwise_or(fg, self.bg)  # 字放到（蓝色）车牌背景中
        com = rot(com, r(10) - 5, com.shape, 5)  # 矩形-->平行四边形
        com = rotRandrom(com, 6, (com.shape[1], com.shape[0]))  # 旋转
        com = tfactor(com)  # 调灰度
        com, loc = random_scene(com, self.noplates_path)  # 放入背景中
        if com is None or loc is None:
            return None, None
        return com, loc

    def gen_batch(self, batchSize, outputPath):
        """
        批量生成图片
        :param batchSize: 设置生成多少张船牌
        :param outputPath: 生成船牌保存位置
        """
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        for i in range(batchSize):
            plate_str = gen_plate_string()
            img, loc = self.generate(plate_str)
            if img is None:
                continue
            cv2.imwrite(outputPath + "/" + str(i) + ".jpg", img)
            with open(outputPath + "/" + 'txt' + ".txt", 'a', encoding='utf-8') as obj:
                line = str(i) + ".jpg" + ' ' + plate_str + '\n'
                obj.write(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_dir', default='background', help='bg_img dir')
    parser.add_argument('--out_dir', default='./plate_train/', help='output dir')
    parser.add_argument('--make_num', default=2500, type=int, help='num')
    return parser.parse_args()


def main(args):
    gen = GenPlateScene("./font/platech.ttf", './font/platechar.ttf', args.bg_dir)
    gen.gen_batch(args.make_num, args.out_dir)


if __name__ == '__main__':
    main(parse_args())
