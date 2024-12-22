import torch
import torch.nn as nn
import torch.nn.functional as F


# Spectral Attention Module
class SpectralAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super(SpectralAttention, self).__init__()
        self.shared_layer_one = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU()
        )
        self.shared_layer_two = nn.Linear(channels // ratio, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Average pooling
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        avg_out = self.shared_layer_one(avg_pool)
        avg_out = self.shared_layer_two(avg_out)
        
        # Max pooling
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(b, c)
        max_out = self.shared_layer_one(max_pool)
        max_out = self.shared_layer_two(max_out)
        
        # Attention
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        return x * attention


# SSAM Block
class SSAMBlock(nn.Module):
    def __init__(self, channels, ratio=8, kernel_size=7):
        super(SSAMBlock, self).__init__()
        self.spectral_attention = SpectralAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        spectral_att = self.spectral_attention(x)
        spatial_att = self.spatial_attention(x)
        return spectral_att + spatial_att / 2


# U-Net with SSAM Block
class SSUNet(nn.Module):
    def __init__(self, inchannels=3, num_classes=1):
        super(SSUNet, self).__init__()
        self.encoder1 = self._block(inchannels, 64, ssam=True)
        self.encoder2 = self._block(64, 128, ssam=True)
        self.encoder3 = self._block(128, 256, ssam=True)
        self.encoder4 = self._block(256, 512, ssam=True)

        self.bottleneck = self._block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self._block(1024, 512, ssam=True)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self._block(512, 256, ssam=True)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self._block(256, 128, ssam=True)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self._block(128, 64, ssam=True)

        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _block(self, in_channels, out_channels, kernel_size=3, dropout=0.2, ssam=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        ]
        if ssam:
            layers.append(SSAMBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        out = self.output_conv(dec1)

        return out


