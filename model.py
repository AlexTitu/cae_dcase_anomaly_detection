import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=4,  # 4x4 kernel
                stride=2,  # 2x upsampling
                padding=1  # keeps output spatial size doubled
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=2048):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = ConvBlock(1, 16)    # [128, 32] 256
        self.enc2 = ConvBlock(16, 32)   # [64, 16] 128
        self.enc3 = ConvBlock(32, 64)   # [32, 8] 46
        self.dropout = nn.Dropout2d(p=0.2)

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(64 * 8 * 4, latent_dim) # [8, 32]

        # ---- Decoder ----
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 4)
        self.dec1 = UpBlock(64, 32)
        self.dec2 = UpBlock(32, 16)
        self.dec3 = UpBlock(16, 16)
        # self.dec4 = UpBlock(8, 8)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Keep same spatial size
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # ---- Encoder ----
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        #x = self.dropout(x)
        # x = self.enc4(x)
        # x = self.dropout(x)
        latent = self.fc_enc(self.flatten(x))

        # ---- Decoder ----
        x = self.fc_dec(latent).view(-1, 64, 8, 4)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        # x = self.dec4(x)
        out = self.sigmoid(self.out_conv(x))
        return out
