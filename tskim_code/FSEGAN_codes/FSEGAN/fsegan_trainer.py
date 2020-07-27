#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

import numpy as np
import matplotlib.pyplot as plt
import os
import params

from tqdm import tqdm

# %%
class FSEGAN_Trainer():
    """Train FSEGAN"""
    def __init__(self, generator, discriminator, classifier,
                gan_criterion, classifier_criterion,
                sample_imgs=None, img_save_path=None,
                device='cuda', w_adversarial = 100, vad_w_adversarial = 0.01,
                writer=None):
        """Train Parameters"""
        self.device = device
        self.w_adversarial = w_adversarial
        self.vad_w_adversarial = vad_w_adversarial
        self.global_step = 0
        self.epoch = 0
        self.writer = writer
        self.sample_imgs = sample_imgs
        self.img_save_path = img_save_path

        """Get models"""
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier

        """Optimizers"""
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_C = optim.Adam(self.classifier.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.gan_criterion = gan_criterion
        self.classifier_criterion = classifier_criterion

    def reset_grads(self):
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()
        self.optimizer_C.zero_grad()
    
    def save_models(self):
        torch.save(self.generator, params.GENERATOR_SAVE_PATH)
        torch.save(self.discriminator, params.DISCRIMINATOR_SAVE_PATH)
        torch.save(self.classifier, params.CLASSIFIER_SAVE_PATH)
    
    def sample_enhance_imgs(self, epoch):
        self.generator.eval()

        #sample_imgs = torch.Tensor(self.sample_imgs)
        sample = self.sample_imgs.float().to(self.device)
        sample_E = self.generator(sample).cpu().detach().numpy()
        for idx, img in enumerate(sample_E):
            img_name = os.path.join(self.img_save_path, 
            "Enhanced_Image" + str(idx+1)+ "_epoch"+ str(epoch) + '.png')
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(sample_E[idx, 0, :, :])
            if self.writer is not None:
                self.writer.add_figure("Enhanced/"+str(idx+1), fig, idx + 1)
            fig.savefig(img_name)
        return sample_E
    
    def test_imgs(self, test_dict, epoch=0):
        keys = list(test_dict.keys())
        self.generator.eval()
        for idx, (snr, noise) in enumerate(keys):
            mel = test_dict[(snr, noise)]
            mel = mel.float().unsqueeze(0).to(self.device)
            generated = self.generator(mel).cpu().detach().numpy()
            _img_name = "snr_{}_noise_{}".format(snr, noise)
            img_name = os.path.join(self.img_save_path,
                "TEST_Image" + _img_name + "_" + str(epoch) + '.png')
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(generated[0, 0, :, :])
            if self.writer is not None:
                self.writer.add_figure("Enhanced/" + _img_name, fig, idx + 1)
            fig.savefig(img_name)
    
    def test_imgs_original(self, test_dict, epoch=0):
        keys = list(test_dict.keys())
        for idx, (snr, noise) in enumerate(keys):
            mel = test_dict[(snr, noise)]
            _img_name = "snr_{}_noise_{}".format(snr, noise)
            img_name = os.path.join(self.img_save_path,
                "TEST_Image" + _img_name + "_" + str(epoch) + '.png')
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(mel[ 0, :, :])
            if self.writer is not None:
                self.writer.add_figure("Orignal/" + _img_name, fig, idx + 1)
            fig.savefig(img_name)

    def train(self, noise_clean_dataloader):
        torch.autograd.set_detect_anomaly(True)
        self.epoch += 1
        d_loss = []
        g_loss = []
        for idx, (noisy, clean) in enumerate(noise_clean_dataloader):
            if clean.size(0) <= 1:
                continue
            self.discriminator.train()
            self.generator.train()
            noisy = noisy.float().to(self.device) # (batch, 1, 128, 128)
            clean = clean.float().to(self.device) # (batch, 1, 128, 128)
            
            #############################
            ##         Train D          #
            #############################
            #Train Discriminator with Clean data & Noisy Data
            self.optimizer_D.zero_grad()
            mixture_CN = torch.cat([clean, noisy], dim=1)
            pred_CN = self.discriminator(mixture_CN) # 0 -> clean, 1 -> noisy, [batch, 1]
            labels = torch.ones((clean.size(0),1)).to(self.device)
            loss_CN = self.criterion(pred_CN, labels) # L2 loss -> to 0

            #Train Discriminator with Generated
            generated = self.generator(noisy)
            generated_D = generated.detach()

            mixture_GN = torch.cat([generated_D, noisy], dim=1) #[2, 128, 128]
            pred_GN = self.discriminator(mixture_GN)
            
            labels = torch.zeros((clean.size(0),1)).to(self.device)
            loss_GN = self.gan_criterion(pred_GN, labels)
            loss_D = 0.5 * (loss_CN + loss_GN)

            #Back-prop
            
            loss_D.backward()
            self.optimizer_D.step()

            #############################
            ##         Train G          #
            #############################
            self.discriminator.eval()
            self.optimizer_G.zero_grad()
            generated = self.generator(noisy)
            mixture_G_GN = torch.cat([generated, noisy], dim=1)
            pred_G_GN = self.discriminator(mixture_G_GN)
            
            #loss from discriminator
            labels = torch.ones((clean.size(0),1)).to(self.device)
            loss_G_D = self.gan_criterion(pred_G_GN, labels)
            #loss from L2 Loss
            loss_G_L = torch.mean(torch.abs(generated - clean)) * self.w_adversarial
            #Total Loss
            loss_G = loss_G_D + loss_G_L

            #Back-prop
            loss_G.backward()
            self.optimizer_G.step()

            d_loss.append(loss_D.item())
            g_loss.append(loss_G.item())
            self.global_step += 1
            if self.global_step % 10 == 0:
                print("EPOCH [{:5d}] Loss d : {:.4f} Loss G D {:.4f} Loss L2 {:.4f}".format(self.epoch, loss_D.item(),
                    loss_G_D.item(), loss_G_L.item()))
                self.writer.add_scalar("Loss/loss_d", loss_D.item(), self.global_step)
                self.writer.add_scalar("Loss/loss_l2", loss_G_L.item(), self.global_step)
                self.writer.add_scalar("loss/loss_g_d", loss_G_D.item(), self.global_step)
        
        return np.mean(np.array(d_loss)), np.mean(np.array(g_loss))

    def train_VAD(self, noise_clean_dataloader):
        torch.autograd.set_detect_anomaly(True)
        self.epoch += 1
        d_loss = []
        g_loss = []
        auc = []
        for idx, ((noisy, clean), vad_label) in enumerate(noise_clean_dataloader):
            if clean.size(0) <= 1:
                continue
            self.optimizer_C.zero_grad()
            self.optimizer_D.zero_grad()
            self.optimizer_G.zero_grad()

            torch.autograd.set_detect_anomaly(True)
            self.discriminator.train()
            self.generator.train()
            self.classifier.train()
            noisy = noisy.float().to(self.device) # (batch, 1, 128, 128)
            clean = clean.float().to(self.device) # (batch, 1, 128, 128)
            vad_label = vad_label.float().to(self.device) # (batch, 128)

            #############################
            ##         Train D          #
            #############################
            #Train Discriminator with Clean data & Noisy Data
            mixture_CN = torch.cat([clean, noisy], dim=1)
            pred_CN = self.discriminator(mixture_CN) # 0 -> clean, 1 -> noisy, [batch, 1]
            labels = torch.ones((clean.size(0),1)).to(self.device)
            loss_CN = self.gan_criterion(pred_CN, labels) # L2 loss -> to 0

            #Train Discriminator with Generated
            generated = self.generator(noisy)
            generated_D = generated.detach()

            mixture_GN = torch.cat([generated_D, noisy], dim=1) #[2, 128, 128]
            pred_GN = self.discriminator(mixture_GN)
            
            labels = torch.zeros((clean.size(0),1)).to(self.device)
            loss_GN = self.gan_criterion(pred_GN, labels)
            loss_D = 0.5 * (loss_CN + loss_GN)

            #Back-prop
            loss_D.backward()
            self.optimizer_D.step()

            #############################
            ##         Train G          #
            #############################
            self.discriminator.eval()
            generated = self.generator(noisy)
            mixture_G_GN = torch.cat([generated, noisy], dim=1)
            pred_G_GN = self.discriminator(mixture_G_GN)
            
            #loss from discriminator
            labels = torch.ones((clean.size(0),1)).to(self.device)
            loss_G_D = self.gan_criterion(pred_G_GN, labels)
            #loss from L2 Loss
            loss_G_L = torch.mean(torch.abs(generated - clean)) * self.w_adversarial
            #Total Loss
            loss_G = loss_G_D + loss_G_L

            #Back-prop
            loss_G.backward(retain_graph=True)
            
            ###########################
            ##          Train C      ##
            ###########################
            vad_pred = self.classifier(generated)
            loss_C = self.classifier_criterion(vad_pred, vad_label) * self.vad_w_adversarial
            loss_C.backward()
            """
            cc = clean.cpu().detach().numpy()[0, 0, :, :]
            ll = vad_label.cpu().detach().numpy()[0, :]
            ll = list(map(lambda x: 50 if x == 1 else 0, ll))
            #print(dd.shape)
            dd = vad_pred.cpu().detach().numpy()[0, :]
            print(dd.shape)
            dd = list(map(lambda x: x * 50, dd))
            plt.imshow(cc)
            plt.plot(ll)
            plt.plot(dd)
            plt.show()
            """

            self.optimizer_C.step()
            self.optimizer_G.step()
            
            l_c = torch.unique(vad_label.cpu().detach().view(-1))
            if len(l_c) > 1:
                auc += [roc_auc_score(y_true=vad_label.cpu().detach().view(-1), y_score=vad_pred.cpu().detach().view(-1))]
            
            d_loss.append(loss_D.item())
            g_loss.append(loss_G.item())
            self.global_step += 1
            if self.global_step % 10 == 0:
                _auc = np.mean(np.array(auc))
                print("EPOCH [{:5d}] AUC : {:.2f} Loss C : {:.4f} Loss d : {:.4f} Loss G D {:.4f} Loss L2 {:.4f}".format(self.epoch, _auc, loss_C.item(), loss_D.item(),
                    loss_G_D.item(), loss_G_L.item()))
                self.writer.add_scalar("AUC/train_auc", _auc, self.global_step)
                self.writer.add_scalar("Loss/loss_d", loss_D.item(), self.global_step)
                self.writer.add_scalar("Loss/loss_l2", loss_G_L.item(), self.global_step)
                self.writer.add_scalar("loss/loss_g_d", loss_G_D.item(), self.global_step)
        
        auc = np.mean(np.array(auc))
        
        return auc, np.mean(np.array(d_loss)), np.mean(np.array(g_loss))
    
    def test(self, dataloader, flag="Librispeech"):
        print("####################################")
        print("           TEST {}".format(flag))
        print("####################################")
        self.generator.eval()
        self.classifier.eval()

        preds = []
        ground_truth = []
        with tqdm(dataloader, ncols=100) as _tqdm:
            for idx, (noise, label) in enumerate(_tqdm):
                noise = noise.float().to(self.device)
                label = label.float().to(self.device)

                generated = self.generator(noise)
                vad_pred = self.classifier(generated)

                preds.append(vad_pred.view(-1).cpu().detach().numpy())
                ground_truth.append(label.view(-1).cpu().detach().numpy())
        
        print("Evaluating ...")
        ground_truth = np.array(ground_truth[:-1])
        ground_truth = np.reshape(ground_truth, (-1,)).astype('int')

        preds = np.array(preds[:-1])
        preds = np.reshape(preds, (-1,))

        auc = roc_auc_score(y_true=ground_truth, y_score=preds)
        print("AUC : ", auc)
        print(classification_report(y_true=ground_truth, y_pred=preds.round(), target_names=['non-speech', 'speech']))
        
        return auc


