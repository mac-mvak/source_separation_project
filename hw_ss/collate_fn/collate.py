import logging
import torch
import torch.nn as nn
from typing import List
from pathlib import Path


logger = logging.getLogger(__name__)


def adder(vec, v):
    if vec is None:
        vec = v
    else:
        size_1, size_2 = vec.shape[-1], v.shape[-1]
        pad = size_1 - size_2
        vec = nn.functional.pad(vec, (0, max(-pad, 0)))
        v = nn.functional.pad(v, (0, max(pad, 0)))
        vec = torch.cat([vec, v])
    return vec


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    ref_audios = None
    mix_audios = None
    target_audios = None
    ref_audios_length = []
    mix_audios_length = []
    target_audios_length = []
    target_ids = []
    noise_ids = []
    snrs = []
    audio_names = []
    for item in dataset_items:
        ref_audios = adder(ref_audios, item['ref_audio'])
        mix_audios = adder(mix_audios, item['mix_audio'])
        target_audios = adder(target_audios, item['target_audio'])
        ref_audios_length.append(item['ref_audio'].shape[-1])
        mix_audios_length.append(item['mix_audio'].shape[-1])
        target_audios_length.append(item['target_audio'].shape[-1])
        target_ids.append(item['target_id'])
        noise_ids.append(item['noise_id'])
        snrs.append(item['snr'])
        audio_names.append(
            Path(item['audio_paths'][0]).name[:-8]
        )
    result_batch = {'ref_audios' : ref_audios, 
                    'mix_audios' : mix_audios,
                    'target_audios' : target_audios,
                    'ref_audios_length': torch.tensor(ref_audios_length, dtype=int),
                    "mix_audios_length" : torch.tensor(mix_audios_length, dtype=int),
                    "target_audios_length": torch.tensor(target_audios_length, dtype=int),
                    "target_ids": torch.tensor(target_ids, dtype=int),
                    "noise_ids" : torch.tensor(noise_ids, dtype=int),
                    "snrs" : snrs,
                    "audio_names" : audio_names
                    }
    return result_batch

