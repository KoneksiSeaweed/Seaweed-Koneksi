import torch
import torch.nn as nn

class unet(nn.Module):
    def __init__(self, inchannels, outclass = 1):
        super().__init__()

        #Encoder/Contraction Block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels= inchannels, out_channels=64, kernel_size=3, padding=1), #output 126x126x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.conv_block11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), #output 124x124x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) #output 62x62x64
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels=128, kernel_size=3, padding=1), #output 60x60x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        
        self.conv_block22 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), #output 58x58x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 29x29x128
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels= 128, out_channels=256, kernel_size=3, padding=1), #output 27x27x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        
        self.conv_block33 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), #output 25x25x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        
        self.maxpool3= nn.MaxPool2d(kernel_size=2, stride=2) #output 12x12x256

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels= 256, out_channels=512, kernel_size=3, padding=1), #output 10x10x512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        
        self.conv_block44 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output 8x8x512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        
        #Decoder/Expansion Block
        self.upconv1 = nn.ConvTranspose2d(in_channels = 512, out_channels=256, kernel_size = 2, stride = 2)
        self.up11 = nn.Sequential(nn.Conv2d(512, 256, kernel_size= 3, padding= 1), nn.ReLU(inplace = True))
        self.up12 = nn.Sequential(nn.Conv2d(256, 256, kernel_size= 3, padding= 1), nn.ReLU(inplace = True))

        self.upconv2 = nn.ConvTranspose2d(in_channels = 256, out_channels=128, kernel_size = 2, stride = 2)
        self.up21 = nn.Sequential(nn.Conv2d(256, 128, kernel_size= 3, padding= 1), nn.ReLU(inplace = True))
        self.up22 = nn.Sequential(nn.Conv2d(128, 128, kernel_size= 3, padding= 1), nn.ReLU(inplace = True))

        self.upconv3 = nn.ConvTranspose2d(in_channels = 128, out_channels=64, kernel_size = 2, stride = 2)
        self.up31 = nn.Sequential(nn.Conv2d(128, 64, kernel_size= 3, padding= 1), nn.ReLU(inplace = True))
        self.up32 = nn.Sequential(nn.Conv2d(64, 64, kernel_size= 3, padding= 1), nn.ReLU(inplace = True))

        # output layer/Classification
        self.outconv = nn.Conv2d(64, outclass, kernel_size=1)

        #Forward
    def forward(self, x):
        #encoder
        outconv1 = self.conv_block1(x)
        outconv11 = self.conv_block11(outconv1) 
        maxconv1 = self.maxpool1(outconv11)
        outconv2 = self.conv_block2(maxconv1)
        outconv22 = self.conv_block22(outconv2)
        maxconv2 = self.maxpool2(outconv22)
        outconv3 = self.conv_block3(maxconv2)
        outconv33 = self.conv_block33(outconv3)
        maxconv3 = self.maxpool3(outconv33)
        outconv4 = self.conv_block4(maxconv3)
        outconv44 = self.conv_block44(outconv4)

        # Decoder
        upconv1 = self.upconv1(outconv44)
        upconv11 = torch.cat([upconv1, outconv33], dim=1)
        updconv11 = self.up11(upconv11)
        updconv12 = self.up12(updconv11)

        upconv2 = self.upconv2(updconv12)
        upconv22 = torch.cat([upconv2, outconv22], dim=1)
        updconv21 = self.up21(upconv22)
        updconv22 = self.up22(updconv21)

        upconv3 = self.upconv3(updconv22)
        upconv33 = torch.cat([upconv3, outconv11], dim=1)
        updconv31 = self.up31(upconv33)
        updconv32 = self.up32(updconv31)

        # Output layer
        out = self.outconv(updconv32)

        return out