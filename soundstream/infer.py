import torch
from lightning import Soundstream
import sys
import soundfile as sf
import resampy

model_checkpoint = sys.argv[1]
inference_data_path = sys.argv[2]

inference_model = Soundstream.load_from_checkpoint(model_checkpoint)


segment_length = 24000
data, _ = sf.read(
    inference_data_path, dtype='float32', always_2d=True)
data = data.mean(axis=1, keepdims=False)
if data.shape[0] != segment_length:
    data = resampy.resample(
                data, data.shape[0], segment_length, axis=0, filter='kaiser_fast')[:segment_length]
data = torch.tensor(data)

output = inference_model.predice(data)

sf.write('answer.wav', data.squeeze(0).detach().numpy(), 24000)
sf.write('output.wav', output.squeeze(0).detach().numpy(), 24000)