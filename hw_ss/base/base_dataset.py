import logging
from typing import List
import torchaudio
from torch.utils.data import Dataset

from hw_ss.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            **kwargs
    ):
        self.config_parser = config_parser
        self.log_spec = config_parser["preprocessing"]["log_spec"]

        self._assert_index_is_valid(index)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        ref_path = data_dict["path_ref"]
        mix_path = data_dict["path_mix"]
        target_path = data_dict["path_target"]
        ref_wave = self.load_audio(ref_path)
        mix_wave = self.load_audio(mix_path)
        target_wave = self.load_audio(target_path)
        return {
            "ref_audio": ref_wave,
            "mix_audio": mix_wave,
            #"duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "target_audio": target_wave,
            "target_id": data_dict["target_id"],
            "noise_id" : data_dict["noise_id"],
            "snr" : data_dict["snr"],
            #"text_encoded": self.text_encoder.encode(data_dict["text"]),
            "audio_paths": [ref_path, mix_path, target_path],
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["target_id"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor


    
    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path_ref" in entry, (
                "Each dataset item should include field 'path_ref'"
                " -path to reference audio"
            )
            assert "path_mix" in entry, (
                "Each dataset item should include field 'path_mix'" " - path to mixed audio."
            )
            assert "path_target" in entry, (
                "Each dataset item should include field 'path_target'"
                " - path to target audio."
            )
