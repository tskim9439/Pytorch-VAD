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

# %%
# 2. Define generator
class dataloader_generator():
    def __init__(self, noise_dict, clean_dict, transform, batch_size=64, window_size = 128, n_data_per_epoch = 4620):
        self.noise_dict = noise_dict
        self.clean_dict = clean_dict
        self.transform = transform
        self.batch_size = batch_size
        self.n_data_per_epoch = n_data_per_epoch
        self.window_size = window_size

        self.noise_keys = list(self.noise_dict.keys())
        random.shuffle(self.noise_keys)
    
    def next_loader(self):
        x = []
        y = []
        while True:
            random.shuffle(self.noise_keys)
            for noise_key in self.noise_keys:
                clean_key = noise_key.split('-')[0]
                
                x.append(self.noise_dict[noise_key])
                y.append(self.clean_dict[clean_key])

                if len(x) >= self.n_data_per_epoch:
                    _noise = _clean = np.concatenate(list(map(utils.preprocess_spec(feature='mel', doskip=False, 
                        window_size=self.window_size + 1), x)), axis=0)
                    _clean = np.concatenate(list(map(utils.preprocess_spec(feature='mel', doskip=False, 
                        window_size=self.window_size + 1), y)), axis=0)
                    
                    dataset = utils.FSEGAN_Dataset(_noise, _clean, transform=self.transform)
                    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                            batch_size = self.batch_size,
                                                            shuffle=True
                                                        )
                    x = []
                    y = []
                    yield data_loader

transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor()])
data_loaders = dataloader_generator(noise_dict=noise_pickle, clean_dict=clean_pickle,
                                    transform=transform, batch_size=BATCH)

sample_loader = next(data_loaders.next_loader())
sample_noisy_imgs, sample_clean_imgs  = next(iter(sample_loader))
sample_noisy_imgs = sample_noisy_imgs[:3]
sample_clean_imgs = sample_clean_imgs[:3]

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
summary(fsegan_generator, (1, 128, 128))
summary(fsegan_discriminator, (2, 128, 128))

#%%
# 4. Train
criterion = nn.MSELoss()
trainer = fsegan_trainer.FSEGAN_Trainer(generator=fsegan_generator, 
                                    discriminator=fsegan_discriminator, 
                                    criterion=criterion,
                                    sample_imgs=sample_noisy_imgs,
                                    img_save_path=IMG_SAVE_PATH,
                                    device=device, writer=writer)
min_loss_g = 999999.
for epoch in range(EPOCH):
    data_loader = next(iter(data_loaders.next_loader()))
    d_loss, g_loss = trainer.train(data_loader)
    
    sample_imgs = trainer.sample_enhance_imgs(epoch + 1)
    writer.add_scalar("LOSS/D_LOSS", d_loss, epoch+1)
    writer.add_scalar("LOSS/G_LOSS", g_loss, epoch+1)
    
    if g_loss <= min_loss_g:
        print("\n#######################################################")
        print("EPOCH [{:3d}] LOSS D {:.4f} LOSS G {:.4f} Model saved".format(epoch + 1, d_loss, g_loss))
        print("#######################################################\n")
        trainer.save_models()
        min_loss_g = g_loss

# %%
