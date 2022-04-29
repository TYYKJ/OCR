import cv2
import numpy as np
from PIL import ImageDraw, Image, ImageFont
from charset_normalizer.utils import is_ascii

from ocr import DetInfer

cap = cv2.VideoCapture('/home/cat/Desktop/船舶号视频录制/鲁蓬渔78788.mp4')


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.avi'):  # 保证文件名的后缀是.avi
            name += '.avi'

        self.__name = name  # 文件名
        self.__height = height  # 高
        self.__width = width  # 宽
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 如果是avi视频，编码需要为MJPG
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()


class Annotator:

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = True
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = ImageFont.truetype("/home/cat/PycharmProjects/OCR/田氏颜体大字库2.0.ttf", 30, encoding="utf-8")
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle((box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1), fill=color)
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def draw_bbox(img_path, result, color=(0, 0, 255), thickness=2):
    import cv2
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    # for point in result:
    # point = point.astype(int)
    cv2.polylines(img_path, [np.int32(result)], True, color, thickness)
    return img_path


infer = DetInfer(det_model_path='weights/DB-dpn68-epoch=79-hmean=0.89-recall=1.00-precision=0.80.ckpt',
                 threshold=0.5, device='cuda:0')
txt = '鲁蓬渔78788'
# width height
vw = VideoWriter('鲁蓬渔78788.avi', 1680, 672)
while True:
    success, im = cap.read()

    if success:
        im = im[160:, :, :]
        print(im.shape)
        # width, height = im.shape
        # print(width, height)

        ann = Annotator(im)
        draw = ImageDraw.Draw(Image.fromarray(im))
        img_points = infer.get_box_points_list(im)
        # print(img_points)

        if img_points:
            for points in img_points:
                left = points[0]
                # right = left + 40
                xyxy = [left[0], left[1], left[0] + 300, left[1] + 300]
                points1 = points.reshape((-1, 1, 2))

                ann.text(xy=np.array(xyxy), text=txt)
                im = ann.result()
                im = draw_bbox(im, points1)

                vw.write(im)
    else:
        break
vw.close()
cv2.destroyAllWindows()
        # cv2.imshow('1', im)
        # cv2.waitKey(1)
