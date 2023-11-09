import torch
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_ss.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ScaleInvariantSignalDistortionRatio()


    def si_sdr(self, est, target, length):
        est, target = est.squeeze()[:length], target.squeeze()[:length]
        if est.shape[-1] < length:
            est = F.pad(est, (0, length - est.shape[-1]))
        return self.metric(est, target)

    def __call__(self, short_decode, target_audios, target_audios_length, **batch):
        anses = torch.empty(target_audios.shape[0], device=target_audios.device)
        for i in range(target_audios.shape[0]):
           
           anses[i] = self.si_sdr(short_decode[i,:], target_audios[i, :], target_audios_length[i])
        return anses.mean()



