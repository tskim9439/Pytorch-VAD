#%%
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from tensorboardX import SummaryWriter

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import params
import utils
import models
import rnn_base_models
import fsegan_trainer
import evaluator
import data_loader
# %%
#####################
## Define Variables #
#####################
EPOCH = params.EPOCH
BATCH = params.BATCH
WRITER_PATH = params.WRITER_PATH
IMG_SAVE_PATH = params.IMG_SAVE_PATH

if not os.path.isdir(WRITER_PATH):
    os.mkdir(WRITER_PATH)

if not os.path.isdir(IMG_SAVE_PATH):
    os.mkdir(IMG_SAVE_PATH)

writer = SummaryWriter(WRITER_PATH)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
###########################
## Load Data & Preprocess #
###########################
# 1. Load Pickle
with open(params.CLEAN_PICKLE, "rb") as f:
    clean_pickle = pickle.load(f)

with open(params.NOISE_PICKLE, "rb") as f:
    noise_pickle = pickle.load(f)

with open(params.LABEL_PICKLE, "rb") as f:
    label_pickle = pickle.load(f)

with open(params.TEST_NOISE_PICKLE, "rb") as f:
    libri_x_pickle = pickle.load(f)

with open(params.TEST_LABEL_PICKLE, "rb") as f:
    libri_y_pickle = pickle.load(f)

#%%
#1-1. Sample librispeech aurora data per SNR and NOISE
test_snrs = params.TEST_SNRS
test_noises = params.TEST_NOISES
libri_sample_imgs = {}
libri_x_keys = list(libri_x_pickle.keys())
libri_x = []
libri_y = []

for idx, key in enumerate(libri_x_keys):
    if idx % 2 == 0:
        snr, noise, clean_key = key.split('/')[0], key.split('/')[1], key.split('/')[2]
        
        mel = libri_x_pickle[key]
        label = libri_y_pickle[key]

        libri_x.append(mel)
        libri_y.append(label)

        if (snr, noise) != libri_sample_imgs and mel.shape[-1] > 128:
            libri_sample_imgs[(snr, noise)] = torch.Tensor(np.log(mel[:,:,:128] + 1e-8))

#%%
# 2. Define generator
transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor()])

libri_x = np.concatenate(list(map(utils.preprocess_spec(feature='mel', doskip=False, 
                                                    window_size=128 + 1), libri_x)), axis=0)[:,:128,:]
libri_y = np.concatenate(
                        list(map(utils.moving_window_label(window_size=128 + 1), libri_y)), axis=0)[:,:128]

libri_dataset = utils.FSEGAN_Libri_Dataset(libri_x, libri_y, transform=transform)

libri_dataloader = torch.utils.data.DataLoader(dataset=libri_dataset, batch_size=BATCH)
#%%
data_loaders = utils.FSEGAN_dataloader_generator(noise_dict=noise_pickle, clean_dict=clean_pickle, label_dict=label_pickle,
                                    transform=transform, batch_size=BATCH)

sample_loader = next(data_loaders.next_loader())

(sample_noisy_imgs, sample_clean_imgs), sample_l  = next(iter(sample_loader))
sample_noisy_imgs = sample_noisy_imgs[:3]
sample_clean_imgs = sample_clean_imgs[:3]

#%%
for idx, img in enumerate(sample_clean_imgs):
    img_name = os.path.join(IMG_SAVE_PATH, "Clean_Image" + str(idx+1) + '.png')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(sample_clean_imgs[idx, 0, :, :])
    if writer is not None:
        writer.add_figure("TARGET/"+str(idx+1), fig, idx + 1)
    fig.savefig(img_name)

for idx, img in enumerate(sample_noisy_imgs):
    img_name = os.path.join(IMG_SAVE_PATH, "Orig_Noisy_Image" + str(idx+1) + '.png')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(sample_noisy_imgs[idx, 0, :, :])
    if writer is not None:
        writer.add_figure("Input_Noisy/"+str(idx+1), fig, idx + 1)
    fig.savefig(img_name)

#%%
# 3. Get Model & Trainer
fsegan_generator = models.FSEGAN_Unet().to(device)
fsegan_discriminator = models.FSEGAN_Discriminator().to(device)
fsegan_classifier = models.FSEGAN_Classifier().to(device)
summary(fsegan_generator, (1, 128, 128))
summary(fsegan_discriminator, (2, 128, 128))
summary(fsegan_classifier, (1, 128, 128))

#%%
# 4. Train
gan_criterion = nn.MSELoss()
vad_criterion = nn.MSELoss()
trainer = fsegan_trainer.FSEGAN_Trainer(generator=fsegan_generator, 
                                    discriminator=fsegan_discriminator,
                                    classifier=fsegan_classifier,
                                    classifier_criterion=vad_criterion,
                                    gan_criterion=gan_criterion,
                                    vad_w_adversarial=1.,
                                    sample_imgs=sample_noisy_imgs,
                                    img_save_path=IMG_SAVE_PATH,
                                    device=device, writer=writer)
trainer.test_imgs_original(libri_sample_imgs)
min_loss_g = 999999.
for epoch in range(EPOCH):
    data_loader = next(iter(data_loaders.next_loader()))
    auc, d_loss, g_loss = trainer.train_VAD(data_loader)
    
    sample_imgs = trainer.sample_enhance_imgs(epoch + 1)
    writer.add_scalar("LOSS/D_LOSS", d_loss, epoch+1)
    writer.add_scalar("LOSS/G_LOSS", g_loss, epoch+1)
    writer.add_scalar("AUC/AUC_EPOCH", auc, epoch + 1)

    if g_loss <= min_loss_g:
        print("\n#######################################################")
        print("EPOCH [{:3d}] AUC : {:.2f} LOSS D {:.4f} LOSS G {:.4f} Model saved".format(epoch + 1, auc, d_loss, g_loss))
        print("#######################################################\n")
        trainer.save_models()
        min_loss_g = g_loss
    
    #Test
    if epoch % 5 == 0:
        test_auc = trainer.test(libri_dataloader)
        writer.add_scalar("TEST/AUC", test_auc, epoch+1)
        trainer.test_imgs(libri_sample_imgs, epoch= epoch + 1)

# %%
