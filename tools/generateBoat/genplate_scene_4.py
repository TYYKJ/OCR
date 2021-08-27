# coding=utf-8
"""
生成船牌数据，将船牌放到自然图像中
"""
import argparse
import os

from PIL import ImageFont

from plate_common import *


def gen_plate_string():
    """生成船牌号码字符串"""
    plate_str = ""
    for cpos in range(9):
        if cpos == 0:
            plate_str += chars[15]
        elif cpos == 1:
            plate_str += chars[65]
        elif cpos == 2:
            plate_str += chars[73]
        elif cpos == 3:
            plate_str += chars[72]
        else:
            plate_str += chars[31 + random_seed(10)]
    return plate_str


class GenPlateScene:
    """船牌数据生成器，船牌放在自然场景中，位置信息存储在txt文件中"""

    def __init__(self, font_ch_path: str, font_en_path: str, bg_folder: str):
        """
        :param font_ch_path: 中文字体
        :param font_en_path: 英文字体
        :param bg_folder: 场景图片文件夹
        """
        self.fontC = ImageFont.truetype(font_ch_path, 43, 0)
        self.fontE = ImageFont.truetype(font_en_path, 26, 0)
        self.img = np.array(Image.new("RGB", (123, 32), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread(TEMPLATE_IMAGE), (123, 32))
        self.no_plates_path = []
        for parent, _, filenames in os.walk(bg_folder):
            for filename in filenames:
                self.no_plates_path.append(parent + "/" + filename)

    def draw(self, boat_num: str) -> np.ndarray:
        """
        设置船牌文字位置

        :param boat_num: 船牌号
        :return:返回合成之后的船牌图片
        """
        offset = 0
        self.img[0:32, offset + 8:offset + 8 + 12] = generate_ch_characters(self.fontC, boat_num[0])
        self.img[0:32, offset + 8 + 12:offset + 8 + 12 + 12] = generate_ch_characters(self.fontC, boat_num[1])
        self.img[0:32, offset + 8 + 12 + 12:offset + 8 + 12 + 12 + 12] = generate_ch_characters(self.fontC, boat_num[2])
        self.img[0:32, offset + 8 + 12 + 12 + 12:offset + 8 + 12 + 12 + 12 + 12] = generate_ch_characters(self.fontC,
                                                                                                          boat_num[3])
        for i in range(5):
            base = offset + 8 + 12 + 12 + 12 + 12 + i * 12
            self.img[0:32, base:base + 12] = generate_en_characters(self.fontE, boat_num[i + 4])
        return self.img

    def generate(self, text: str) -> tuple:
        """
        对船牌号文字和图片进行旋转变形等操作

        :param text:生成的船牌号
        :return: 图像和位置信息
        """
        print(text, len(text))
        fg = self.draw(text.encode(encoding="utf-8").decode(encoding="utf-8"))
        fg = cv2.bitwise_not(fg)
        com = cv2.bitwise_or(fg, self.bg)
        com = rot(com, random_seed(10) - 5, com.shape, 5)
        com = image_distortion(com, 6, (com.shape[1], com.shape[0]))
        com = change_gray_and_color(com)
        com, loc = random_scene(com, self.no_plates_path)
        if com is None or loc is None:
            return None, None
        return com, loc

    def gen_batch(self, batch_size, output_path):
        """
        批量生成图片

        :param batch_size: 设置生成多少张船牌
        :param output_path: 生成船牌保存位置
        """
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for i in range(batch_size):
            plate_str = gen_plate_string()
            img, loc = self.generate(plate_str)
            if img is None:
                continue
            if plate_str[-5:] + ".jpg" not in os.listdir(output_path):
                cv2.imwrite(output_path + "/" + plate_str[-5:] + ".jpg", img)
                # cv2.imencode('.jpg', img)[1].tofile(outputPath + "/" + plate_str[-5:] + ".jpg")
                with open(output_path + "/" + 'txt' + ".txt", 'a', encoding='utf-8') as obj:
                    line = plate_str[-5:] + ".jpg" + ' ' + plate_str + '\n'
                    obj.write(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_dir', default='background', help='bg_img dir')
    parser.add_argument('--out_dir', default='./plate_train/', help='output dir')
    parser.add_argument('--make_num', default=2000, type=int, help='num')
    return parser.parse_args()


def main(args):
    gen = GenPlateScene("./font/platech.ttf", './font/platechar.ttf', args.bg_dir)
    gen.gen_batch(args.make_num, args.out_dir)


if __name__ == '__main__':
    img_folder = './images'
    for filenames in os.listdir(img_folder):
        TEMPLATE_IMAGE = os.path.join(img_folder,filenames)
        main(parse_args())
