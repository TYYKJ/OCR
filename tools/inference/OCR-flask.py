from flask import Flask, request, jsonify
from gevent import monkey;
import yaml

monkey.patch_all()
from gevent.pywsgi import WSGIServer
import numpy as np
import json
from tools.inference.inference import ManagedBertModel, Inference as Model
from service_streamer import Streamer, ThreadedStreamer

app = Flask(__name__)
model = None
streamer = None


@app.route("/stream", methods=["POST"])
def stream_predict():
    data = request.get_json()
    if data:
        pass
    else:
        data = request.get_data()
        data = json.loads(data)
    inList = []
    inputs = np.array(data["inputs"]).astype("uint8")
    savePath = data["savePath"]
    inList.append([inputs, savePath])
    outputs = streamer.predict(inList)
    return jsonify(outputs)


if __name__ == "__main__":
    stream = open('flask.yaml', mode='r', encoding='utf-8')
    data = yaml.load(stream, Loader=yaml.FullLoader)

    model = Model(det_model_path=data['det_model_path'],
                  rec_model_path=data['rec_model_path'],
                  angle_model_path=data['angle_model_path'], device=data['device'],
                  dict_path=data['dict_path'],
                  classify_classes=data['classes'], std=0.5, mean=0.5, threshold=0.7)

    streamer = ThreadedStreamer(model.predict, batch_size=64, max_latency=0.1)  # 单GPU卡
    # streamer = Streamer(ManagedBertModel, batch_size=64, max_latency=0.1, worker_num=4, cuda_devices=data['cuda_devices']) # 多GPU卡

    app.run(port=5005, debug=False)
    WSGIServer(("0.0.0.0", 5005), app).serve_forever()
