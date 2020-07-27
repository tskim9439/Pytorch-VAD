#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
from tqdm import tqdm

import torch
import torchvision
import torchaudio

import params

# %%
snrs = params.SNRS
noises = params.NOISES
base_path = params.PATH
n_mels = params.N_MELS
n_fft = params.N_FFT
sample_rate = params.SAMPLE_RATE
clean_path = "/data/datasets/ai_challenge/TIMIT_extended/train_wav_10/"

to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
downsampling = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate)

#%%
def get_fp(fp_list, path):
    sub_dirs = os.listdir(path)
    for sub_dir in sub_dirs:
        if sub_dir in snrs:
            fp_list = get_fp(fp_list, os.path.join(path, sub_dir))
        elif sub_dir in noises:
            fp_list += glob(os.path.join(path, sub_dir) + "/train_wav/*.wav")     
    return fp_list

#%%
def save_clean():
    des = ''
    clean_dict = {}
    noisy_wav_list = get_fp([], base_path)
    clean_wav_list = glob("/data/datasets/ai_challenge/TIMIT_extended/train_wav_10/*.wav")
    with tqdm(clean_wav_list, ncols=100, desc=des) as _tqdm:
        for clean_fp in _tqdm:
            name = clean_fp.split('/')[-1][:-4]
            des = name
            clean_data, sr = torchaudio.load(clean_fp)
            down_sampled = downsampling(clean_data)
            mel_spec = to_mel(down_sampled)
            mel_np = mel_spec.numpy()
            clean_dict[name] = mel_np
    with open(os.path.join(base_path, "clean_dict_128_mel.pickle"), "wb") as f:
        pickle.dump(clean_dict, f)
#save_clean()

#%%
from multiprocessing import Process

noisy_dict = {}
def save_noise_128_mel():
    print("function start")
    noisy_wav_list = get_fp([], base_path)
    six = int(len(noisy_wav_list) / 6)
    noisy_dict = {}
    with tqdm(noisy_wav_list, ncols=100) as _tqdm:
        for noisy in _tqdm:
            noise_fp = noisy.split('/')[-1][:-4] + "-" + noisy.split('/')[-4]
            noisy_data, sr = torchaudio.load(noisy)
            downsampled = downsampling(noisy_data)
            mel_spec = to_mel(downsampled)
            mel_np = mel_spec.numpy()

            noisy_dict[noise_fp] = mel_np
        
        with open(os.path.join(base_path, "noisy_dict_128_mel.pickle"), "wb") as f:
            pickle.dump(noisy_dict, f)
#save_noise_128_mel()
#%%
def save_noise_128_mel_1(array):
    print("function start")
    #noisy_wav_list = get_fp([], base_path)
    #six = int(len(noisy_wav_list) / 6)
    with tqdm(array, ncols=100, desc="HI") as _tqdm:
        print("start")
        for noisy in start:
            noise_fp = noisy.split('/')[-1][:-4] + "-" + noisy.split('/')[-4]
            noisy_data, sr = torchaudio.load(noisy)
            downsampled = downsampling(noisy_data)
            mel_spec = to_mel(downsampled)
            mel_np = mel_spec.numpy()

            noisy_dict[noise_fp] = mel_np


# %%
