import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import pickle
import os
import random


EPSILON = 1e-8
LOG_EPSILON = np.log(EPSILON)

PATH = '/data/datasets/ai_challenge/' # inside a docker container

SOURCEPATH = "/data/datasets/ai_challenge/ST_attention_dataset"
TARGETPATH = "/data/datasets/ai_challenge/TIMIT_noisex3"
TESTPATH = "/data/datasets/ai_challenge/TIMIT_NOISEX_extended/TEST"

tail = '_no_noise_aug.pickle'


def sequence_to_windows(sequence, 
                        pad_size, 
                        step_size, 
                        skip=1,
                        padding=True, 
                        const_value=0):
    '''
    SEQUENCE: (time, ...)
    PAD_SIZE:  int -> width of the window // 2
    STEP_SIZE: int -> step size inside the window
    SKIP:      int -> skip windows...
        ex) if skip == 2, total number of windows will be halved.
    PADDING:   bool -> whether the sequence is padded or not
    CONST_VALUE: (int, float) -> value to fill in the padding

    RETURN: (time, window, ...)
    '''
    assert (pad_size-1) % step_size == 0

    window = np.concatenate([np.arange(-pad_size, -step_size, step_size),
                             np.array([-1, 0, 1]),
                             np.arange(step_size+1, pad_size+1, step_size)],
                            axis=0)
    window += pad_size
    output_len = len(sequence) if padding else len(sequence) - 2*pad_size
    window = window[np.newaxis, :] + np.arange(0, output_len, skip)[:, np.newaxis]

    if padding:
        pad = np.ones((pad_size, *sequence.shape[1:]), dtype=np.float32)
        pad *= const_value
        sequence = np.concatenate([pad, sequence, pad], axis=0)

    return np.take(sequence, window, axis=0)


def windows_to_sequence(windows,
                        pad_size,
                        step_size):
    windows = np.array(windows)
    sequence = np.zeros((windows.shape[0],), dtype=np.float32)
    indices = np.arange(1, windows.shape[0]+1)
    indices = sequence_to_windows(indices, pad_size, step_size, True)

    for i in range(windows.shape[0]):
        pred = windows[np.where(indices-1 == i)]
        sequence[i] = pred.mean()
    
    return sequence


def pad(spec, pad_size, axis, const_value):
    padding = np.ones((*spec.shape[:axis], pad_size, *spec.shape[axis+1:]),
                      dtype=np.float32)
    padding *= const_value
    return np.concatenate([padding, spec, padding], axis=axis)


# TODO (test is required)
def preprocess_spec(feature='mel', skip=1, doskip=True, window_size=7, overlap=0.5):
    if feature not in ['spec', 'mel', 'mfcc']:
        raise ValueError(f'invalid feature - {feature}')

    def _preprocess_spec(spec):
        if feature in ['spec', 'mel']:
            spec = np.log(spec + EPSILON)
        if feature == 'mel':
            spec = (spec - 4.5252) / 2.6146 # normalize
        spec = spec[0,:,:]
        spec = spec.transpose(1, 0) # to (time, freq)
        if doskip:
            windows = sequence_to_windows(spec, 
                                        pad_size=19, step_size=9,
                                        skip=skip, padding=True, const_value=LOG_EPSILON)
        else:
            windows = moving_window(spec, window_size=window_size, overlap=overlap)
        return windows
    return _preprocess_spec

def normalize_mel(spec):
    spec = np.log(spec + EPSILON)
    spec = (spec - 4.5252) / 2.6146
    return spec


# TODO (test is required)
def label_to_window(skip=1):
    def _preprocess_label(label):
        label = sequence_to_windows(
            label, 19, 9, skip, True)
        return label
    return _preprocess_label

def moving_window(sequence, window_size=7, overlap = 0.5):
    t_len = len(sequence)
    window_half_len = int(math.ceil(window_size / 2))
    step_size = int(window_size * overlap)

    idx = np.arange(window_half_len, t_len - window_half_len, step_size)
    window = np.arange(-window_half_len + 1, window_half_len)

    windows_idx = window[np.newaxis, :] + idx[:, np.newaxis]

    return np.take(sequence, windows_idx, axis=0)

def moving_window_label(window_size=7, overlap=0.5):
    def _moving_window_label(label):
        label = moving_window(label, window_size=window_size, overlap=overlap)
        return label
    return _moving_window_label

class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self, x, y, transform=None):
    self.x_data = x
    self.y_data = y
    
    if self.x_data.shape[0] > self.y_data.shape[0]:
        self.y_data = self.y_data[:self.x_data.shape[0], :]
    
    if self.x_data.shape[0] < self.y_data.shape[0]:
        self.x_data = self.x_data[:self.y_data.shape[0], :]
    
    print("x shape : {}  y shape : {}".format(self.x_data.shape, self.y_data.shape))
    self.transform = transform

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx, :, :]
    y = self.y_data[idx]

    if self.transform is not None:
        x = self.transform(x)
        y = self.transform(x)
    
    x = x[0, :, :]
    x = np.transpose(x, (1, 0))

    return x, y

class FSEGAN_Dataset(torch.utils.data.Dataset): 
  def __init__(self, x, y, transform=None):
    #self.x_data = x[:, None, :-1, :]
    #self.y_data = y[:, None, :-1, :]
    self.x_data = x
    self.y_data = y
    
    print("x shape : {}  y shape : {}".format(self.x_data.shape, self.y_data.shape))
    self.transform = transform

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx, :, :]
    y = self.y_data[idx, :, :]

    if self.transform is not None:
        x = self.transform(x)
        y = self.transform(y)
        
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)

    return x, y

class FSEGAN_Libri_Dataset(torch.utils.data.Dataset): 
  def __init__(self, x, y, transform=None):
    #self.x_data = x[:, None, :-1, :]
    #self.y_data = y[:, None, :-1, :]
    self.x_data = x
    self.y_data = y
    
    print("x shape : {}  y shape : {}".format(self.x_data.shape, self.y_data.shape))
    self.transform = transform

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx, None, :, :]
    y = torch.Tensor(self.y_data[idx, :])

    if self.transform is not None:
        x = self.transform(x)
        
    x = x.transpose(0, 1)

    return x, y


class fsegan_vad_dataloader_generator():
    def __init__(self, noise_dict, clean_dict, label_dict, transform, batch_size=64, window_size = 128, n_data_per_epoch = 4620):
        self.noise_dict = noise_dict
        self.clean_dict = clean_dict
        self.label_dict = label_dict

        self.transform = transform
        self.batch_size = batch_size
        self.n_data_per_epoch = n_data_per_epoch
        self.window_size = window_size

        self.noise_keys = list(self.noise_dict.keys())
        random.shuffle(self.noise_keys)
    
    def next_loader(self):
        x = []
        y = []
        lab = []
        while True:
            random.shuffle(self.noise_keys)
            for noise_key in self.noise_keys:
                clean_key = noise_key.split('-')[0]
                
                x.append(self.noise_dict[noise_key])
                y.append(self.clean_dict[clean_key])
                lab.append(self.label_dict[clean_key])

                if len(x) >= self.n_data_per_epoch:
                    _noise = _clean = np.concatenate(list(map(preprocess_spec(feature='mel', doskip=False, 
                        window_size=self.window_size + 1), x)), axis=0)
                    _clean = np.concatenate(list(map(preprocess_spec(feature='mel', doskip=False, 
                        window_size=self.window_size + 1), y)), axis=0)
                    _label = np.concatenate(
                        list(map(moving_window_label(window_size=self.window_size + 1), lab)), axis=0)
                    
                    dataset = FSEGAN_VAD_Dataset(_noise, _clean, _label, transform=self.transform)
                    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                            batch_size = self.batch_size,
                                                            shuffle=True
                                                        )
                    x = []
                    y = []
                    lab = []
                    yield data_loader

class FSEGAN_VAD_Dataset(torch.utils.data.Dataset): 
  def __init__(self, x, y, lab, transform=None):
    self.x_data = x[:, None, :-1, :]
    self.y_data = y[:, None, :-1, :]
    self.label = lab
    
    print("x shape : {}  y shape : {} label shape : {}".format(self.x_data.shape,
                        self.y_data.shape, self.label.shape))
    self.transform = transform

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx, :, :, :]
    y = self.y_data[idx, :, :, :]
    l = self.label[idx, :128]

    if self.transform is not None:
        x = self.transform(x)
        y = self.transform(y)
        
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)

    return (x, y), l

class FSEGAN_dataloader_generator():
    def __init__(self, noise_dict, clean_dict, label_dict, transform, batch_size=64, window_size = 128, n_data_per_epoch = 4620):
        self.noise_dict = noise_dict
        self.clean_dict = clean_dict
        self.label_dict = label_dict

        self.transform = transform
        self.batch_size = batch_size
        self.n_data_per_epoch = n_data_per_epoch
        self.window_size = window_size

        self.noise_keys = list(self.noise_dict.keys())
        random.shuffle(self.noise_keys)
    
    def next_loader(self):
        x = []
        y = []
        lab = []
        while True:
            random.shuffle(self.noise_keys)
            for noise_key in self.noise_keys:
                clean_key = noise_key.split('-')[0]

                clean_mel = self.clean_dict[clean_key]
                noise_mel = self.noise_dict[noise_key]
                label = self.label_dict[clean_key]

                if label.shape[0] < clean_mel.shape[-1]:
                    clean_mel = clean_mel[:,:,:label.shape[0]]
                    noise_mel = noise_mel[:,:,:label.shape[0]]
                elif label.shape[0] > clean_mel.shape[-1]:
                    label = label[:clean_mel.shape[-1]]
                
                x.append(noise_mel)
                y.append(clean_mel)
                lab.append(label)

                if len(x) >= self.n_data_per_epoch:
                    _noise = _clean = np.concatenate(list(map(preprocess_spec(feature='mel', doskip=False, 
                        window_size=self.window_size + 1), x)), axis=0)
                    _clean = np.concatenate(list(map(preprocess_spec(feature='mel', doskip=False, 
                        window_size=self.window_size + 1), y)), axis=0)
                    _label = np.concatenate(
                        list(map(moving_window_label(window_size=self.window_size + 1), lab)), axis=0)
                    
                    dataset = FSEGAN_VAD_Dataset(_noise, _clean, _label, transform=self.transform)
                    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                            batch_size = self.batch_size,
                                                            shuffle=True
                                                        )
                    x = []
                    y = []
                    lab = []
                    yield data_loader