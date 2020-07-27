#%%
import os
import numpy as np
#%%
"""Parameters related to data"""
PATH = "/data/datasets/ai_challenge/ST_attention_dataset"
CLEAN_PICKLE = os.path.join(PATH, "clean_dict_128_mel.pickle")
NOISE_PICKLE = os.path.join(PATH, "noisy_dict_128_mel.pickle")
LABEL_PICKLE = os.path.join(PATH, "timit_label_8khz_mel.pickle")

TEST_NOISE_PICKLE = os.path.join(PATH, "libri_val_8khz_mel_128_x.pickle")
TEST_LABEL_PICKLE = os.path.join(PATH, "libri_val_8khz_mel_128_y.pickle")

SNRS = ['-10', '-5', '0', '5', '10']
NOISES = ['f16', 'factory1.1.0.raw', 'factory2',
 'machinegun', 'volvo', 'destroyerengine', 'm109', 'babble']

TEST_SNRS = [-10, -5, 0]
TEST_NOISES = ['street', 'airport', 'restaurant', 'car', 'exhibition', 'babble']

N_MELS = 128
N_FFT = 512
SAMPLE_RATE = 8000

"""Parameters Related to Train"""
EPOCH = 10000
BATCH = 64
WRITER_PATH = os.path.join(os.getcwd(), "fsegan_adv_v3")
IMG_SAVE_PATH = os.path.join(os.getcwd(), "fsegan_imgs_adv_v3")
GENERATOR_SAVE_PATH = os.path.join(os.getcwd(), "fsegan_generator_adv_v3.h5")
DISCRIMINATOR_SAVE_PATH = os.path.join(os.getcwd(), "fsegan_discriminator_adv_v3.h5")
CLASSIFIER_SAVE_PATH = os.path.join(os.getcwd(), "fsegan_cassifier_adv_v3.h5")
