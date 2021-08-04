# coding=utf-8

import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from math import *

INDEX_PROVINCE = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9,
                  "苏": 10, "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19,
                  "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29,
                  "新": 30}

INDEX_DIGIT = {"0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40}
INDEX_LETTER = {"A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47,
                "H": 48, "J": 49, "K": 50, "L": 51, "M": 52, "N": 53,
                "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58,
                "U": 59, "V": 60, "W": 61, "X": 62, "Y": 63, "Z": 64}
INDEX_CT = {"烟": 65, "牟": 66, "福": 67, "蓬": 68, "长": 69, "海": 70, "渔": 71
            }

PLATE_CHARS_PROVINCE = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                        "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
                        "新"}
PLATE_CHARS_DIGIT = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
PLATE_CHARS_LETTER = {"A", "B", "C", "D", "E", "F", "G",
                      "H", "J", "K", "L", "M", "N",
                      "P", "Q", "R", "S", "T",
                      "U", "V", "W", "X", "Y", "Z"}
PLATE_CT = {"烟", "牟", "福", "蓬", "长", "海", "渔"}

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64, "烟": 65, "牟": 66, "福": 67, "蓬": 68, "长": 69, "海": 70, "渔": 71}

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z", "烟", "牟", "福", "蓬", "长", "海", "渔"]


def rot(img, angel, shape, max_angel):
    """
    对图片进行旋转
    :param img:输入图片
    :param angel:角度
    :param shape:图片大小
    :param max_angel:最大角度
    :return:返回变化后的图片
    """
    size_o = [shape[1], shape[0]]
    size = (shape[1] + int(shape[0] * sin((float(max_angel) / 180) * 3.14)), shape[0])
    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))
    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
    if angel > 0:
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)

    return dst


def rotRandrom(img, factor, size):
    """
    仿射畸变
    :param img:输入图片
    :param factor:畸变的参数
    :param size:图片尺寸
    :return:返回变化后的图片
    """
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)],
                       [r(factor), shape[0] - r(factor)],
                       [shape[1] - r(factor), r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst


def tfactor(img):
    """
    调灰度，调颜色
    :param img: 输入图片
    :return: 变化后的图片
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + np.random.random() * 0.8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def random_scene(img, data_set):
    """
    将船牌放入自然场景图像中，并返回该图像和位置信息
    :param img: 输入图片
    :param data_set: 自然场景图片
    :return: 图片和位置
    """
    bg_img_path = data_set[r(len(data_set))]
    print(len(data_set))
    env = cv2.imread(bg_img_path)
    if env is None:
        print(bg_img_path, 'is not a good file')
        return None, None
    if env.shape[1] <= 130 or env.shape[0] <= 32:
        print(env.shape)
        return None, None
    new_height = img.shape[0]
    new_width = img.shape[1]
    new_width = int(new_width)
    new_height = int(new_height)
    img = cv2.resize(img, (new_width, new_height))
    if env.shape[1] <= img.shape[1] or env.shape[0] <= img.shape[0]:
        print(env.shape, '---', img.shape)
        return None, None
    x = r(env.shape[1] - img.shape[1])
    y = r(env.shape[0] - img.shape[0])
    bak = (img == 0)
    bak = bak.astype(np.uint8) * 255
    inv = cv2.bitwise_and(bak, env[y:y + new_height, x:x + new_width, :])
    img = cv2.bitwise_or(inv, img)
    env[y:y + new_height, x:x + new_width, :] = img[:, :, :]
    return env, (x, y, x + img.shape[1], y + img.shape[0])


def GenCh(f, val):
    """
    生成汉字
    :param f:字体
    :param val:文本内容
    :return:图片矩阵
    """
    img = Image.new("RGB", (45, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3), val, (0, 0, 0), font=f)
    img = img.resize((18, 32))
    A = np.array(img)
    return A


def GenCh1(f, val):
    """
    生成英文字符
    :param f:字体
    :param val:文本内容
    :return:图片矩阵
    """
    img = Image.new("RGB", (12, 32), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), val.encode(encoding="utf-8").decode('utf-8'), (0, 0, 0), font=f)
    A = np.array(img)
    return A


def r(val):
    return int(np.random.random() * val)
