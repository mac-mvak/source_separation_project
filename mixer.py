import os
import glob
from glob import glob

import numpy as np

import random
import json

import librosa
from pathlib import Path
import torchaudio
from tqdm import tqdm
import soundfile as sf
import pyloudnorm as pyln
from speechbrain.utils.data_utils import download_file
import shutil
import matplotlib.pyplot as plt

import torch
from hw_ss.utils import ROOT_PATH

from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")



URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
}


def snr_mixer(clean, noise, snr):
    amp_noise = np.linalg.norm(clean) / 10**(snr / 20)

    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise

    mix = clean + noise_norm

    return mix

def vad_merge(w, top_db):
    intervals = librosa.effects.split(w, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

def cut_audios(s1, s2, sec, sr):
    cut_len = sr * sec
    len1 = len(s1)
    len2 = len(s2)

    s1_cut = []
    s2_cut = []

    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])

        segment += 1

    return s1_cut, s2_cut

def fix_length(s1, s2, min_or_max='max'):
    # Fix length
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:  # max
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2


def create_mix(idx, triplet, snr_levels, out_dir, test=False, sr=16000, **kwargs):
    trim_db, vad_db = kwargs["trim_db"], kwargs["vad_db"]
    audioLen = kwargs["audioLen"]

    s1_path = triplet["target"]
    s2_path = triplet["noise"]
    ref_path = triplet["reference"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]
    norm_target_id = triplet["norm_target_id"]
    norm_noise_id = triplet["norm_noise_id"]

    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    ref, _ = sf.read(os.path.join('', ref_path))

    meter = pyln.Meter(sr) # create BS.1770 meter

    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    loudsRef = meter.integrated_loudness(ref)

    s1Norm = pyln.normalize.loudness(s1, louds1, -20)
    s2Norm = pyln.normalize.loudness(s2, louds2, -20)
    refNorm = pyln.normalize.loudness(ref, loudsRef, -20.0)

    amp_s1 = np.max(np.abs(s1Norm))
    amp_s2 = np.max(np.abs(s2Norm))
    amp_ref = np.max(np.abs(refNorm))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0:
        return

    if trim_db:
        ref, _ = librosa.effects.trim(refNorm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1Norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2Norm, top_db=trim_db)

    if len(ref) < sr:
        return

    path_mix = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-mixed.wav")
    path_target = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-target.wav")
    path_ref = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-ref.wav")
    path_json = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + ".json")

    snr = np.random.choice(snr_levels, 1).item()

    if not test:
        s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        s1_cut, s2_cut = cut_audios(s1, s2, audioLen, sr)

        for i in range(len(s1_cut)):
            json_dict = {"snr": snr,
                         "target_id": norm_target_id,
                         "noise_id": norm_noise_id}
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)

            louds1 = meter.integrated_loudness(s1_cut[i])
            s1_cut[i] = pyln.normalize.loudness(s1_cut[i], louds1, -20.0)
            loudMix = meter.integrated_loudness(mix)
            mix = pyln.normalize.loudness(mix, loudMix, -20.0)

            path_mix_i = path_mix.replace("-mixed.wav", f"_{i}-mixed.wav")
            path_target_i = path_target.replace("-target.wav", f"_{i}-target.wav")
            path_ref_i = path_ref.replace("-ref.wav", f"_{i}-ref.wav")
            path_json_i = path_json.replace(".json", f"_{i}.json")
            json_dict["path_mix"] = path_mix_i
            json_dict["path_target"] = path_target_i
            json_dict["path_ref"] = path_ref_i
            sf.write(path_mix_i, mix, sr)
            sf.write(path_target_i, s1_cut[i], sr)
            sf.write(path_ref_i, ref, sr)
            with open(path_json_i, 'w') as f:
                json.dump(json_dict, f)

    else:
        json_dict = {"snr": snr,
                         "target_id": norm_target_id,
                         "noise_id": norm_noise_id}
        s1, s2 = fix_length(s1, s2, 'max')
        mix = snr_mixer(s1, s2, snr)
        louds1 = meter.integrated_loudness(s1)
        s1 = pyln.normalize.loudness(s1, louds1, -20.0)

        loudMix = meter.integrated_loudness(mix)
        mix = pyln.normalize.loudness(mix, loudMix, -20.0)
        json_dict["path_mix"] = path_mix
        json_dict["path_target"] = path_target
        json_dict["path_ref"] = path_ref

        sf.write(path_mix, mix, sr)
        sf.write(path_target, s1, sr)
        sf.write(path_ref, ref, sr)
        with open(path_json, 'w') as f:
            json.dump(json_dict, f)

class LibriSpeechSpeakerFiles:
    def __init__(self, speaker_id, speaker_dict, audios_dir, audioTemplate="*-norm.wav"):
        self.id = speaker_id
        self.norm_id = speaker_dict[speaker_id]
        self.files = []
        self.audioTemplate=audioTemplate
        self.files = self.find_files_by_worker(audios_dir)

    def find_files_by_worker(self, audios_dir):
        speakerDir = os.path.join(audios_dir,self.id) #it is a string
        chapterDirs = os.scandir(speakerDir)
        files=[]
        for chapterDir in chapterDirs:
            files = files + [file for file in glob(os.path.join(speakerDir,chapterDir.name)+"/"+self.audioTemplate)]
        return files

class MixtureGenerator:
    def __init__(self, speakers_files, out_folder, nfiles=5000, test=False, randomState=42, name="train"):
        self.speakers_files = speakers_files # list of SpeakerFiles for every speaker_id
        self.nfiles = nfiles
        self.randomState = randomState
        self.name = name
        self.out_folder = out_folder
        self.test = test
        random.seed(self.randomState)
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def generate_triplets(self):
        i = 0
        all_triplets = {"reference": [], "target": [], "noise": [], "target_id": [], "noise_id": [],
                         "norm_target_id": [], "norm_noise_id" : []}
        while i < self.nfiles:
            spk1, spk2 = random.sample(self.speakers_files, 2)

            if len(spk1.files) < 2 or len(spk2.files) < 2:
                continue

            target, reference = random.sample(spk1.files, 2)
            noise = random.choice(spk2.files)
            all_triplets["reference"].append(reference)
            all_triplets["target"].append(target)
            all_triplets["noise"].append(noise)
            all_triplets["target_id"].append(spk1.id)
            all_triplets["noise_id"].append(spk2.id)
            all_triplets["norm_target_id"].append(spk1.norm_id)
            all_triplets["norm_noise_id"].append(spk2.norm_id)
            i += 1

        return all_triplets

    def triplet_generator(self, target_speaker, noise_speaker, number_of_triplets):
        max_num_triplets = min(len(target_speaker.files), len(noise_speaker.files))
        number_of_triplets = min(max_num_triplets, number_of_triplets)

        target_samples = random.sample(target_speaker.files, k=number_of_triplets)
        reference_samples = random.sample(target_speaker.files, k=number_of_triplets)
        noise_samples = random.sample(noise_speaker.files, k=number_of_triplets)

        triplets = {"reference": [], "target": [], "noise": [],
                    "target_id": [target_speaker.id] * number_of_triplets, "noise_id": [noise_speaker.id] * number_of_triplets}
        triplets["target"] += target_samples
        triplets["reference"] += reference_samples
        triplets["noise"] += noise_samples

        return triplets
    
    def make_final_json(self):
        jsons = sorted(glob(os.path.join(self.out_folder, '*.json')))
        final_dicts = []
        for file in jsons:
            with open(file) as f:
                json_dict = json.load(f)
            final_dicts.append(json_dict)
        path = os.path.split(self.out_folder)[0] + f"/{self.name}_index.json"
        with open(path, "w") as f:
            json.dump(final_dicts, f, indent=2)
        
        for file in jsons:
            os.remove(file)
        

    def generate_mixes(self, snr_levels=[0], num_workers=10, update_steps=10, **kwargs):

        triplets = self.generate_triplets()

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = []

            for i in range(self.nfiles):
                triplet = {"reference": triplets["reference"][i],
                           "target": triplets["target"][i],
                           "noise": triplets["noise"][i],
                           "target_id": triplets["target_id"][i],
                           "noise_id": triplets["noise_id"][i],
                           "norm_target_id":  triplets["norm_target_id"][i],
                           "norm_noise_id" : triplets["norm_noise_id"][i]}

                futures.append(pool.submit(create_mix, i, triplet,
                                           snr_levels, self.out_folder,
                                           test=self.test, **kwargs))

            for i, future in enumerate(futures):
                future.result()
                if (i + 1) % max(self.nfiles // update_steps, 1) == 0:
                    print(f"Files Processed | {i + 1} out of {self.nfiles}")
        self.make_final_json()

def normalized_numbers(speakers_list):
    ans = {}
    for i, speaker in enumerate(speakers_list):
        ans[speaker] = i
    return ans

class LibriSpeechDownloader:
    def __init__(self, type="train"):
        if type == "train":
            self.part = "train-clean-100"
        else:
            self.part = "dev-clean"
        data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(self.part)
    
    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
    
    def path(self):
        return self._data_dir / self.part


path_train = LibriSpeechDownloader("train").path()
#path_val = LibriSpeechDownloader("test").path()



path_mixtures_train = ROOT_PATH / "data/datasets/mixture/train"
path_mixtures_train.mkdir(exist_ok=True, parents=True)
#path_mixtures_val = ROOT_PATH / "data/datasets/mixture/val"
#path_mixtures_val.mkdir(exist_ok=True, parents=True)


speakersTrain = [el.name for el in os.scandir(path_train)]
train_speakers_dict = normalized_numbers(speakersTrain)
#speakersVal = [el.name for el in os.scandir(path_val)]
#val_speakers_dict = normalized_numbers(speakersVal)


speakers_files_train = [LibriSpeechSpeakerFiles(i, train_speakers_dict, path_train, audioTemplate="*.flac") for i in speakersTrain][:101]
speakers_files_val = [LibriSpeechSpeakerFiles(i, val_speakers_dict, path_val, audioTemplate="*.flac") for i in speakersVal]

mixer_train = MixtureGenerator(speakers_files_train,
                                path_mixtures_train,
                                nfiles=20000,
                                test=False, name="train")

#mixer_val = MixtureGenerator(speakers_files_val,
#                                path_mixtures_val,
#                                nfiles=200,
#                                test=True, name="val")


mixer_train.generate_mixes(snr_levels=[0],
                           num_workers=8,
                           update_steps=100,
                           trim_db=20,
                           vad_db=20,
                           audioLen=3)

#mixer_val.generate_mixes(snr_levels=[0],
#                           num_workers=8,
#                           update_steps=10,
##                           trim_db=20,
#                           vad_db=20,
#                           audioLen=3)
##

