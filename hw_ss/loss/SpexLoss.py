import torch
import torch.nn as nn
from torch import Tensor



class SISDRLoss(nn.Module):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        super().__init__()

    def si_sdr(self, est, target, length):
        est, target = est.squeeze()[:length], target.squeeze()[:length]
        if est.shape[-1] < length:
            est = nn.functional.pad(est, (0, length - est.shape[-1]))
        alpha = (target * est).sum() / torch.norm(target)**2
        return 20 * torch.log10(torch.norm(alpha * target) / (torch.norm(alpha * target - est) + 1e-6) + 1e-6)


    
    def forward(self, short_decode, medium_decode, long_decode, target_audios, target_audios_length, **batch):
        anses = torch.zeros(target_audios.shape[0], device=target_audios.device)
        for i in range(target_audios.shape[0]):
           anses[i] -= (1 - self.alpha - self.beta) * self.si_sdr(short_decode[i,:], target_audios[i, :], target_audios_length[i])
           anses[i] -= self.alpha * self.si_sdr(medium_decode[i,:], target_audios[i, :], target_audios_length[i])
           anses[i] -= self.beta * self.si_sdr(long_decode[i,:], target_audios[i, :], target_audios_length[i])
        return anses.mean()


class SpexLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        self.sisdr_loss = SISDRLoss(alpha, beta)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, **batch):
        sdr = self.sisdr_loss(**batch)
        ce = self.ce_loss(batch['class_lin'], batch['target_ids'])
        return sdr + self.gamma * ce







