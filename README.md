# TransYNet
Novel architecture for segmentation of Optical Coherence Tomopgrahpy images

The present repository contains the code for the **TransYNet**, a novel architecture developed to segment Optical Coherence Tomography (OCT) scans. It combines the capabilities of two different models, the Y-Net and the TransUNet, that are both based on the standard U-Net architecture. 

### Spatial Encoder
The  encoding branch consists of one 7*7 convolution (stride 2) and 3 encoding blocks, each being composed of a downsampling convolution followed by a triple convolution (1x1 (stride 1), dilated 3x3 (stride 2), 1x1 (stride 1). The block’s output serves two purposes: it is (I) used as a skip-connection to the decoder branch, and (II) downsampled by max pooling (stride 2) before entering the next encoder block. At the bottom, called the bottleneck, the last encoder feature map is double convolved and the depth of the feature maps is doubled (channels * 16). 

## Spectral Encoder
The spectral encoder branch consists of four Fast Fourier Convolution (FFC) blocks. Within each block, a Fourier unit performs an inverse Fourier transform to convert the features back to the spatial domain to facilitate the convergence of the branches. This output is then sent to the decoder, similar as in the spatial encoder. An in-depth description of the archi- tecture can be found in [the original paper](https://arxiv.org/abs/2204.07613).

# Transformer Bottleneck
At the end of the encoding branch, image patches (of size 1x1) tokenized from the last encoder feature map are fed to the transformer to extract global context. The transformer output is then convolved (3x3, stride 1), normalized and passed to the lowest decoding block along with the output of the last encoding block. 

## Decoder
The decoder branch is divided into four blocks that consist of two 3x3 convolutional layers and ReLU functions. Before each block, the input is upsampled by a 2x2 convolution and combined with the skip-connection from the corresponding encoder block. In the last block, a 1x1 convolutional operation is added to create the output segmentation map[13].

TransUNet. The TransUNet implementation was the same as proposed in the original paper. The architecture of TransUNet is comprised of a U-Net whose encoding branch consists of one 7*7 convolution (stride 2) and 3 en- coding blocks, but is augmented by a vision transformer. The encoding blocks differ from those of the basic U-Net, in that triple convolutions (1x1 (stride 1), dilated 3x3 (stride 2), 1x1 (stride 1) are applied after downsampling using a convolution operation instead of pooling. The decoding blocks consist of bilear upsampling by factor 2 and a double convolution (3x3, stride 1). 

