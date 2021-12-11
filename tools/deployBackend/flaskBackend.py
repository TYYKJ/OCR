import os

import cv2
import numpy as np
import yaml
from flask import Flask, request, jsonify
from service_streamer import ThreadedStreamer, Streamer

from tools.inference import Inference

app = Flask(__name__)
model = None
streamer = None


@app.route("/ocr/", methods=["POST"], strict_slashes=False)
def stream_predict():
    upload_file = request.files['file']
    img = upload_file.stream.read()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    result = model.infer(
        img=img,
        img_save_name=request.values.get('saveName'),
        cut_image_save_path=config['cut_image_save_path'],
        need_angle=config['need_angle'],
        need_object=config['need_object']
    )
    return jsonify({'status': 1, 'result': result})


if __name__ == "__main__":
    if os.path.exists('config.yaml'):
        config = open('config.yaml', mode='r', encoding='utf-8')
        config = yaml.load(config, Loader=yaml.FullLoader)

        model = Inference(
            det_model_path=config['det_model_path'],
            rec_model_path=config['rec_model_path'],
            device=config['device'],
            dict_path=config['dict_path'],
            classify_classes=config['classes'], std=0.5, mean=0.5, threshold=0.7,
            angle_classes=['0', '180'],
            angle_classify_model_path=config['angle_model_path'],
            object_classes=None,
            object_classify_model_path=None
        )
        # start child thread as worker
        if len(config['device']) > 1:
            streamer = Streamer(model.infer, batch_size=64, max_latency=0.1, cuda_devices=tuple(config['device']),
                                worker_num=len(config['device']))
        else:
            streamer = ThreadedStreamer(model.infer, batch_size=64, max_latency=0.1)

        app.run(port=5005, debug=True)
    else:
        raise FileNotFoundError('must have a config.yaml file!')
