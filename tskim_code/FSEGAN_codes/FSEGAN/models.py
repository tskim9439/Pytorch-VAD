#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# %%
class FSEGAN_Unet(nn.Module):
    """Define decoder model of FSEGAN"""
    def __init__(self, input_shape=(128, 128)):
        super(FSEGAN_Unet, self).__init__()
        self.encoder = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                    ])
        self.decoder = nn.ModuleList([nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1)
                    ])
    
    def forward(self, x):
        out_feature = []
        encoded = x
        for layer in self.encoder:
            encoded = layer(encoded)
            if isinstance(layer, nn.Conv2d):
                out_feature.append(encoded)
        
        decoded = encoded
        encode_feature_idx = 2
        for idx, layer in enumerate(self.decoder):
            decoded = layer(decoded)

            if isinstance(layer, nn.ConvTranspose2d) and encode_feature_idx <= len(out_feature):
                decoded = torch.cat([decoded, out_feature[-encode_feature_idx]], axis=1)
                encode_feature_idx += 1
        return decoded
#%%
class FSEGAN_Discriminator(nn.Module):
    """Define discriminator of FSEGAN"""
    def __init__(self):
        super(FSEGAN_Discriminator, self).__init__()
        self.bottle_neck = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=(1, 8), stride=1),
            nn.ReLU()
        )
        self.decision = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.bottle_neck(x)
        x = x.view(-1, 8)
        x = self.decision(x)
        x = self.sig(x)
        return x
#%%
class FSEGAN_Classifier(nn.Module):
    """Define decoder model of FSEGAN"""
    def __init__(self, input_shape=(128, 128)):
        super(FSEGAN_Classifier, self).__init__()
        bottle_neck = [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding= 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                    ]
        decision = [
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        ]
        self.bottle_neck = nn.Sequential(*bottle_neck)
        self.decision = nn.Sequential(*decision)
    
    def forward(self, x):
        x = self.bottle_neck(x)
        x = x.view(x.shape[0], -1)
        x = self.decision(x)
        return x
