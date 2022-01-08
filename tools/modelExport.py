import pytorch_lightning as pl
import torch
from ocr import DBDetModel


def onnx_export(model_path):
    filepath = "model.onnx"
    model = DBDetModel.load_from_checkpoint(model_path)
    model.eval()
    input_sample = torch.randn((1, 3, 640, 640))
    model.to_onnx(filepath, input_sample, export_params=True)


onnx_export('')
