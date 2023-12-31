# TransYNet
#### A novel architecture for segmentation of Optical Coherence Tomopgraphy images

The present repository contains the code for the [**TransYNet**](/nets/transynet.py), a novel architecture developed to segment Optical Coherence Tomography (OCT) scans. It combines the capabilities of two different models, the [Y-Net](https://arxiv.org/abs/2204.07613) and the [TransUNet](https://arxiv.org/abs/2102.04306), that are both based on the standard U-Net architecture. The Y-net allows encoding of spectral features within OCT images, while the Transformer encoder allows to better capture global context in the images. The architecture looks as follows: 

![TransYNet architecture](TransYNet_architecture.png "TransYNet architecture")

### Spatial Encoder
The  encoding branch consists of one 7*7 convolution (stride 2) and 3 encoding blocks, each being composed of a downsampling convolution followed by a triple convolution (1x1 (stride 1), dilated 3x3 (stride 2), 1x1 (stride 1). The block’s output serves two purposes: it is (I) used as a skip-connection to the decoder branch, and (II) downsampled by max pooling (stride 2) before entering the next encoder block. At the bottom, called the bottleneck, the last encoder feature map is double convolved and the depth of the feature maps is doubled (channels * 16). 

### Spectral Encoder
The spectral encoder branch consists of four Fast Fourier Convolution (FFC) blocks. Within each block, a Fourier unit performs an inverse Fourier transform to convert the features back to the spatial domain to facilitate the convergence of the branches. This output is then sent to the decoder, similar as in the spatial encoder. An in-depth description of the architecture can be found in [the original paper](https://arxiv.org/abs/2204.07613).

### Transformer Bottleneck
At the end of the encoding branch, image patches (of size 1x1) tokenized from the last encoder feature map are fed to the transformer to extract global context. The transformer output is then convolved (3x3, stride 1), normalized and is then concatenated with the output of the last encoder blocks. By doing so, the information from the last encoder blocks is preserved, and augmented by the global context provided by the transformer. This concatenated tensor is then once again convolved before being fed to the last decoder block.

### Decoder
The decoder branch is divided into four blocks that consist of two 3x3 convolutional layers and ReLU functions. Before each block, the input is upsampled by a 2x2 convolution and combined with the skip-connection from the corresponding encoder blocks. In the last block, a 1x1 convolutional operation is added to create the output segmentation map.

TransYNet is written in PyTorch. 

#### References: 

Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., ... & Zhou, Y. (2021). Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.

Farshad, A., Yeganeh, Y., Gehlbach, P., & Navab, N. (2022, September). Y-Net: A spatiospectral dual-encoder network for medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 582-592). Cham: Springer Nature Switzerland.




