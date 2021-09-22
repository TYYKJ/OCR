import torch

from rec import CRNN

model = CRNN.load_from_checkpoint('//BoatNumber/ocr_rec/inference/ocr_rec.pt')

script = model.to_torchscript()
torch.jit.save(script, 'model_rec.ts')
