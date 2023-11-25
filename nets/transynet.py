import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from nets.ffc import FFC_BN_ACT, ConcatTupleLayer

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


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=2,
            groups=1,
            padding=1,
            dilation=1,
            bias=False,
        )
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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, stride=1, padding=1), # *2 for smoother downsampling
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.layer(x)
        return x


# Defined a single spatial encoder class to be independent of the Transformer
class SpatialEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # self.encoder1 = EncoderBlock(in_channels, out_channels, stride=2)
        self.encoder2 = EncoderBlock(out_channels, out_channels * 2, stride=2)
        self.encoder3 = EncoderBlock(out_channels * 2, out_channels * 4, stride=2)
        self.encoder4 = EncoderBlock(out_channels * 4, out_channels * 4, stride=2)

    def forward(self, x):
    
        # x1 = self.encoder1(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)
    
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        return x4, x3, x2, x1


class SpectralEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ratio_in = 0.5,
    ):
        
        super().__init__()
        
        self.ratio_in = ratio_in
        self.encoder1_f = FFC_BN_ACT(in_channels, out_channels, kernel_size=1, stride = 2, ratio_gin=0, ratio_gout=ratio_in)
        self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_f = FFC_BN_ACT(out_channels, out_channels * 2, kernel_size=1, stride = 1, ratio_gin=ratio_in, ratio_gout=ratio_in)  # was 1,2
        self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_f = FFC_BN_ACT(out_channels * 2, out_channels * 4, kernel_size=1, stride = 1,ratio_gin=ratio_in, ratio_gout=ratio_in)
        self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_f = FFC_BN_ACT(out_channels * 4, out_channels * 4, kernel_size=1, stride = 1,ratio_gin=ratio_in, ratio_gout=ratio_in)  # was 8
        self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.catLayer = ConcatTupleLayer()
        
    def forward(self, x):
        spec_x1 = self.encoder1_f(x)
        
        spec_x1_l, spec_x1_g = spec_x1
        spec_x2 = self.encoder2_f((self.pool1_f(spec_x1_l), self.pool1_f(spec_x1_g)))
        spec_x1 = self.catLayer((spec_x1_l, spec_x1_g)) # Concatenate local and global features for skip connections

        spec_x2_l, spec_x2_g = spec_x2
        spec_x3 = self.encoder3_f((self.pool2_f(spec_x2_l), self.pool2_f(spec_x2_g)))
        spec_x2 = self.catLayer((spec_x2_l, spec_x2_g)) # Concatenate local and global features for skip connections

        spec_x3_l, spec_x3_g = spec_x3
        spec_x4 = self.encoder4_f((self.pool3_f(spec_x3_l), self.pool3_f(spec_x3_g)))
        spec_x3 = self.catLayer((spec_x3_l, spec_x3_g)) # Concatenate local and global features for skip connections

        spec_x4_l, spec_x4_g = spec_x4
        spec_x4 = self.catLayer((spec_x4_l, spec_x4_g))
        

        return spec_x4, spec_x3, spec_x2, spec_x1


# Bringing it all together
class TransformerBottleneck(nn.Module):
        def __init__(
            self,
            img_dim = 256,
            out_channels = 128,
            head_num = 4,
            mlp_dim = 512,
            block_num = 8,
            patch_dim = 16,
            dropout_prob = 0.1,
            skip_prob = 0
        ):
            super().__init__()
            
            self.catLayer = ConcatTupleLayer()
            self.relu = nn.ReLU(inplace=True)
            
            # Vision Transformer
            self.vit_img_dim = img_dim // patch_dim
            self.vit = ViT(
                self.vit_img_dim,
                out_channels * 4 * 2, # * 2 to account for concatenation of the spatial and spectral features
                out_channels * 4 * 2, 
                head_num,
                mlp_dim,
                block_num,
                patch_dim = 1, # Still have to figure out how to deal with the patch size - 1 is a value from the GitHub repo, that I don't understand
                classification = False,
                dropout_prob = dropout_prob,
                skip_prob = skip_prob
            )
            
            # Convolutions after Transformer to increase dimensionality to combine with bottleneck
            self.conv_trans = nn.Conv2d(out_channels * 8, out_channels * 16, kernel_size=3, stride=1, padding=1)
            self.norm_trans = nn.BatchNorm2d(out_channels * 16)

            # Bottleneck convolutions to bottleneck spatial and spectral features in parallel to Transformer, doubling the number of channels
            self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=out_channels * 8, out_channels = out_channels * 8, kernel_size = 3, padding = 1, bias = False),
                                            nn.BatchNorm2d(num_features=out_channels * 8),
                                            nn.ReLU(inplace=True), 
                                            nn.Conv2d(in_channels = out_channels * 8, out_channels = out_channels * 16, kernel_size = 3, padding = 1, bias = False),
                                            nn.BatchNorm2d(num_features=out_channels * 16),
                                            nn.ReLU(inplace=True))
            
            ## Convolutions for common Transformer and encoder output, halving the number of channels
            self.trans_encoder_conv = nn.Conv2d((out_channels * 16) * 2, out_channels * 16, kernel_size=3, stride=1) 
            self.trans_encoder_norm = nn.BatchNorm2d(out_channels * 16)
            
            # Upsampling for Decoder, halving number of channels
            self.upconv = nn.ConvTranspose2d(out_channels * 16, out_channels * 8, kernel_size=3, stride=1)
            self.upsample_transencoder = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

            
        def forward(self, spat_x4, spec_x4):

            combined_encoder = torch.cat((spat_x4, spec_x4), 1)

            x = self.vit(combined_encoder)
            x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

            # Convolutions after Transformer to reduce dimensionality
            x = self.conv_trans(x)
            x = self.norm_trans(x)
            x = self.relu(x)
              
            # Bottleneck spatial and spectral features in parallel to Transformer
            bottleneck = self.bottleneck(combined_encoder)

            ## Concatenate transformer output with combined encoder output
            assert x.shape == bottleneck.shape, f"Dimensions of Transformer output ({x.shape})and bottleneck output ({bottleneck.shape}) do not match."
            trans_encoder = torch.cat((x, bottleneck), dim=1)
            
            ## Convolutions for to combine common Transformer and encoder output
            trans_encoder = self.trans_encoder_conv(trans_encoder)
            trans_encoder = self.trans_encoder_norm(trans_encoder)
            trans_encoder = self.relu(trans_encoder)
            
            # Upsampling for Decoder
            trans_encoder = self.upconv(trans_encoder)
                        

            return trans_encoder  
                
            
class Decoder(nn.Module):
    def __init__(self, out_channels, num_classes):
        super().__init__()

        self.decoder4 = DecoderBlock((out_channels * 8) * 2, out_channels * 4) # We start with 16 * c - 8 from combined bottleneck and transformer, 4 from spatial and 4 from spectral encoder
        self.decoder3 = DecoderBlock((out_channels * 4) * 3, out_channels * 4) # Then: 12 * C; 4 from spatial, 4 from spectral, 4 from decoder block before
        self.decoder2 = DecoderBlock((out_channels * 4) * 2, out_channels * 2)
        self.decoder1 = DecoderBlock((out_channels * 2) * 2, out_channels)
        self.conv_final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        

    def forward(self, trans_encoder, spat_x4, spat_x3, spat_x2, spat_x1, spec_x4, spec_x3, spec_x2, spec_x1):
        
        encoding_comb = torch.cat((spat_x4, spec_x4), dim=1)
        dec4 = torch.cat((trans_encoder, encoding_comb), dim=1)
        dec3 = self.decoder4(dec4)
        
        encoding_comb = torch.cat((spat_x3, spec_x3), dim=1)
        dec3 = torch.cat((dec3, encoding_comb), dim=1)
        dec2 = self.decoder3(dec3)

        encoding_comb = torch.cat((spat_x2, spec_x2), dim=1)
        dec2 = torch.cat((dec2, encoding_comb), dim=1)
        dec1 = self.decoder2(dec2)
        
        encoding_comb = torch.cat((spat_x1, spec_x1), dim=1)
        dec1 = torch.cat((dec1, encoding_comb), dim=1)
        x = self.decoder1(dec1)
        
        y = self.conv_final(x)
        
        return y


class TransYNet(nn.Module):
    def __init__(
        self,
        img_dim = 256,
        in_channels = 3,
        out_channels = 128,
        head_num = 4,
        mlp_dim = 512,
        block_num = 8,
        patch_dim = 16,
        num_classes = 4,
        dropout_prob = 0.1,
        skip_prob = 0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.spatial_encoder = SpatialEncoder(in_channels, out_channels)
        self.spectral_encoder = SpectralEncoder(in_channels, out_channels)
        self.transformer_bottleneck = TransformerBottleneck(img_dim = img_dim, out_channels=out_channels, head_num=head_num, mlp_dim=mlp_dim, block_num=block_num, patch_dim=patch_dim, dropout_prob=dropout_prob, skip_prob=skip_prob)
        self.decoder = Decoder(out_channels, num_classes)

    def forward(self, x):
        spat_x4, spat_x3, spat_x2, spat_x1 = self.spatial_encoder(x)
        spec_x4, spec_x3, spec_x2, spec_x1  = self.spectral_encoder(x)
        
        trans_encoder = self.transformer_bottleneck(spat_x4, spec_x4)
        
        x = self.decoder(trans_encoder, spat_x4, spat_x3, spat_x2, spat_x1, spec_x4, spec_x3, spec_x2, spec_x1)

        return x
    



