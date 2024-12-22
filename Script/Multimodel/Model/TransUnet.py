import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projection and reshape for multi-head attention
        qkv = self.qkv_layer(x)  # Shape: [batch_size, seq_len, embed_dim * 3]
        qkv = qkv.view(batch_size, seq_len, 3, self.head_num, embed_dim // self.head_num)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, batch_size, head_num, seq_len, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        # Equivalent to torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk
        energy = torch.matmul(query, key.transpose(-2, -1)) * self.dk  # Shape: [batch_size, head_num, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attention = F.softmax(energy, dim=-1)

        # Weighted sum of values
        # Equivalent to torch.einsum("... i j , ... j d -> ... i d", attention, value)
        x = torch.matmul(attention, value)  # Shape: [batch_size, head_num, seq_len, head_dim]

        # Reshape back to the original embedding dimension
        x = x.permute(0, 2, 1, 3).contiguous()  # Shape: [batch_size, seq_len, head_num, head_dim]
        x = x.view(batch_size, seq_len, embed_dim)  # Shape: [batch_size, seq_len, embed_dim]

        # Final linear projection
        x = self.out_attention(x)
        return x

class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # Split image into patches
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height // self.patch_dim, self.patch_dim, width // self.patch_dim, self.patch_dim)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [batch_size, num_patches_x, num_patches_y, patch_dim, patch_dim, channels]
        img_patches = x.view(batch_size, -1, self.token_dim)  # [batch_size, num_tokens, token_dim]

        # Project patches to embedding dimension
        project = self.projection(img_patches)  # Shape: [batch_size, num_tokens, embedding_dim]

        # Expand cls_token for each batch and concatenate with the patches
        token = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embedding_dim]
        patches = torch.cat([token, project], dim=1)  # Shape: [batch_size, num_tokens + 1, embedding_dim]

        # Add positional embedding
        patches += self.embedding[:patches.size(1), :]

        # Apply dropout and transformer encoder
        x = self.dropout(patches)
        x = self.transformer(x)

        # Output for classification or otherwise
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x

# first part (encoder)
class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x

class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        x = self.vit(x)
        
        # Reshape `x` from (batch_size, num_tokens, channels) to (batch_size, channels, height, width)
        batch_size, num_tokens, channels = x.size()
        x = x.permute(0, 2, 1).contiguous()  # (batch_size, channels, num_tokens)
        x = x.view(batch_size, channels, self.vit_img_dim, self.vit_img_dim)  # (batch_size, channels, height, width)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3

# second part (decoder)
class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x

class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x
'''
if __name__ == '__main__':
    transunet = TransUNet(img_dim=128,
                          in_channels=4,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)

    print(sum(p.numel() for p in transunet.parameters()))
    print(transunet(torch.randn(1, 4, 128, 128)).shape)
'''
