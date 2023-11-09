import torch
import torch.nn.functional as F
import torchaudio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


metric = PerceptualEvaluationSpeechQuality(16000, mode='wb')

target, _ = torchaudio.load('/home/mac-mvak/code_disk/hw2/samples/1018_101_005738_0-target.wav')
pred, _ = torchaudio.load('/home/mac-mvak/code_disk/hw2/samples/6.wav')

res1 = metric(pred, target)
res2 = metric(pred/5, target)
print(res1, res2)

