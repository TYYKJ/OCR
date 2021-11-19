from det_infer import DetInfer
from rec_infer import RecInfer
# import argparse
# from line_profiler import LineProfiler
from memory_profiler import profile
from det.utils.vis import draw_ocr_box_txt
import numpy as np


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    points = points.astype(np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class OCRInfer(object):
    def __init__(self, det_path, rec_path, dict_path, rec_batch_size=16):
        super().__init__()
        self.det_model = DetInfer(det_path)
        # self.rec_model = RecInfer(rec_path, dict_path, rec_batch_size)

    def do_predict(self, img):
        box_list, score_list = self.det_model.predict(img)
        if len(box_list) == 0:
            return [], [], img
        draw_box_list = [tuple(map(tuple, box)) for box in box_list]
        imgs = [get_rotate_crop_image(img, box) for box in box_list]
        for index, im in enumerate(imgs):
            cv2.imwrite(f'{index}.jpg', im)
        # texts = self.rec_model.predict(imgs)
        # texts = [txt[0][0] for txt in texts]
        # print(texts)
        # debug_img = draw_ocr_box_txt(img, draw_box_list, texts)
        return box_list, score_list, debug_img

    def predict(self, img):
        return self.do_predict(img)


# def init_args():
#     import argparse
#     parser = argparse.ArgumentParser(description='OCR infer')
#     parser.add_argument('--det_path', required=True, type=str, help='det model path')
#     parser.add_argument('--rec_path', required=True, type=str, help='rec model path')
#     parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
#     parser.add_argument('--rec_batch_size', type=int, help='rec batch_size', default=16)
#     parser.add_argument('-time_profile', action='store_true', help='enable time profile mode')
#     parser.add_argument('-mem_profile', action='store_true', help='enable memory profile mode')
#     args = parser.parse_args()
#     return vars(args)


if __name__ == '__main__':
    import cv2

    img = cv2.imread('/home/data/PyCharmProjects/torch-ocr-dev/test_data/1a.jpg')[130:-130, :, :]
    model = OCRInfer(
        det_path='weights/DB-epoch=126-hmean=0.69.ckpt',
        rec_path='/home/data/PyCharmProjects/torch-ocr-dev/tools/train/weights/DB-epoch=107-hmean=0.62.ckpt',
        rec_batch_size=1,
        dict_path='dict.txt'
    )
    txts, boxes, debug_img = model.predict(img)

    h, w, _, = debug_img.shape
    raido = 1
    if w > 1200:
        raido = 600.0 / w
    # debug_img = cv2.resize(debug_img, (int(w * raido), int(h * raido)))

    cv2.imwrite('result.jpg', debug_img)
    # cv2.imshow("debug", debug_img)
    # cv2.namedWindow("debug", cv2.WINDOW_FULLSCREEN)
    # cv2.waitKey()
