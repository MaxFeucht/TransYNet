from einops import rearrange, repeat
import torch
from torch import nn
import numpy as np

# Vision Transformer

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(
            rearrange(qkv, "b t (d k h) -> k b h t d", k=3, h=self.head_num)
        )
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, dropout_prob):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, dropout_prob, p_b):
        super().__init__()

        # Regularization: 
        self.p_b = p_b # Default to 0, i.e., no stochastic depth
        self.dropout_prob = dropout_prob # Default to 0.1

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim, dropout_prob)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x, stochastic_bypass_discount):

        # Add stochastic depth
        if torch.rand(1) < self.p_b * stochastic_bypass_discount:  # Later layers get a higher bypass probability
            return x # Skip layer
            
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num = 12, dropout_prob = 0.1, skip_prob = 0):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embedding_dim, head_num, mlp_dim, dropout_prob, skip_prob)
                for _ in range(block_num)
            ]
        )

    def forward(self, x):
        for i, layer_block in enumerate(self.layer_blocks):
            stochastic_bypass_discount = (i / len(self.layer_blocks)) # Later layers get a higher bypass probability
            x = layer_block(x, stochastic_bypass_discount)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_dim,
        in_channels,
        embedding_dim,
        head_num,
        mlp_dim,
        block_num,
        patch_dim,
        classification = True,
        num_classes = 4,
        dropout_prob = 0.1,
        skip_prob = 0
    ):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim**2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        self.transformer = TransformerEncoder(
            embedding_dim = embedding_dim, 
            head_num = head_num, 
            mlp_dim = mlp_dim, 
            block_num = block_num, 
            dropout_prob = dropout_prob, 
            skip_prob = skip_prob
        )

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(
            x,
            "b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)",
            patch_x=self.patch_dim,
            patch_y=self.patch_dim,
        )

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        token = repeat(
            self.cls_token, "b ... -> (b batch_size) ...", batch_size=batch_size
        )

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[: tokens + 1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x


## TransUNet Parts

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


class Encoder(nn.Module):
    def __init__(self, 
                 img_dim, 
                 in_channels, 
                 out_channels, 
                 head_num, 
                 mlp_dim, 
                 block_num, 
                 patch_dim, 
                 dropout_prob = 0.1, 
                 skip_prob = 0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, 
                       out_channels * 8, 
                       out_channels * 8,
                       head_num,
                       mlp_dim,
                       block_num,
                       patch_dim = 1, # Still have to figure out how to deal with the patch size - 1 is a value from the GitHub repo, that I don't understand
                       classification = False,
                       dropout_prob = dropout_prob,
                       skip_prob = skip_prob) # WATCH OUT - HERE patch_dim = 1, not 16. Not sure if intended or Error in Github repo!

        self.conv2 = nn.Conv2d(out_channels * 8, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels * 4)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, num_classes):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), num_classes, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x

# Putting it all together

class TransUNet(nn.Module):
    def __init__(self, 
                 img_dim, 
                 in_channels, 
                 out_channels, 
                 head_num, 
                 mlp_dim, 
                 block_num, 
                 patch_dim, 
                 num_classes, 
                 dropout_prob = 0.1, 
                 skip_prob = 0):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = Encoder(img_dim, 
                               in_channels, 
                               out_channels,
                               head_num, 
                               mlp_dim, 
                               block_num, 
                               patch_dim, 
                               dropout_prob, 
                               skip_prob)

        self.decoder = Decoder(out_channels, num_classes)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x
