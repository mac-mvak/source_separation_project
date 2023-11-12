import torch
import torch.nn as nn
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio



class SISDRLoss(nn.Module):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        super().__init__()
        self.metric = ScaleInvariantSignalDistortionRatio(zero_mean=True)

    
    def forward(self, short_decode, medium_decode, long_decode, target_audios, **batch):
        short_decode = short_decode.squeeze()
        medium_decode = medium_decode.squeeze()
        long_decode = long_decode.squeeze()
        target_audios = target_audios.squeeze()
        ans = -(1 - self.alpha - self.beta) * self.metric(short_decode, target_audios)
        ans -= self.alpha * self.metric(medium_decode, target_audios)
        ans -= self.beta * self.metric(long_decode, target_audios)
        return ans


class SpexLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        self.sisdr_loss = SISDRLoss(alpha, beta)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, **batch):
        sdr = self.sisdr_loss(**batch)
        if batch['is_train']:
            ce = self.ce_loss(batch['class_lin'], batch['target_ids'])
        else:
            ce = 0
        return sdr + self.gamma * ce







