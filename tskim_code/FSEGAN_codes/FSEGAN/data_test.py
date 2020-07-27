#%%
import torch
import torch.nn as nn
import torchaudio

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
import random
from tqdm import tqdm

import params
# %%
TEST_PATH = "/data/datasets/ai_challenge/ST_attention_Libri_aurora"
SNRS = [-10, -5, 0]
NOISES = ['street', 'airport', 'restaurant', 'car', 'exhibition', 'babble']

# %%
label_fp = "/data/datasets/ai_challenge/ST_attention_Libri_aurora/libri_label_8khz_mel.pickle"
with open(label_fp, "rb") as f:
    label_pickle = pickle.load(f)

fp_keys = list(label_pickle.keys())
random.shuffle(fp_keys)
fp_keys = list(map(lambda x : x[:-9], fp_keys))
fp_keys = fp_keys[:1000]

# %%
mel_dict = {}
label_dict = {}
to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_fft=512, n_mels=128)
down_sample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
_tail = "-rvad-ext"
for snr in SNRS:
    snr_fp = os.path.join(TEST_PATH, str(snr))
    for noise in NOISES:
        noise_fp = os.path.join(snr_fp, noise)

        with tqdm(fp_keys, ncols=100) as _tqdm:
            for idx, key in enumerate(_tqdm):
                _key = "./" + str(snr) + "/" + noise + "/wav/" + key + "-" + noise + ".wav"
                new_key = str(snr) + "/" + noise + "/" + key
                label_key = key + _tail
                data_fp = os.path.join(noise_fp, "wav")
                data_fp = os.path.join(data_fp, key + '-' + noise + ".wav")
                
                wav_data, sr = torchaudio.load(data_fp)
                down_sampled = down_sample(wav_data)
                
                mel = to_mel(down_sampled).numpy()
                label = label_pickle[label_key]
                
                if label.shape[0] > mel.shape[-1]:
                    label = label[:mel.shape[-1]]
                elif label.shape[0] < mel.shape[-1]:
                    mel = mel[:,:,:label.shape[0]]

                mel_dict[new_key] = mel
                label_dict[new_key] = label

# %%
save_fp = "/data/datasets/ai_challenge/ST_attention_dataset"
with open(os.path.join(save_fp, "libri_val_8khz_mel_128_x.pickle"), "wb") as f:
    pickle.dumps(mel_dict, f)

with open(os.path.join(save_fp, "libri_val_8khz_mel_128_y.pickle"), "wb") as f:
    pickle.dumps(label_dict, f)   
# %%
