# Generate Flower Pictures by MLP, AE, VAE, and Diffusion.

1. 实现MLP Autoencoder，Convolutional Autoencoder，Variational Autoencoder以及Diffsion，用于flowers图像的压缩与采样生成任务。
   
2. 为了正常运行代码，请正常安装torch与torchvision依赖，安装说明见[PyTorch官方页面](https://pytorch.org/get-started/locally/)。

## Background: Datasets

本项目提供了一个简易的数据集，存放在`./flowers`文件夹下。(full dataset可以从 [这里](https://www.kaggle.com/datasets/l3llff/flowers) 获取)。

原数据集包含**15,700张**来自**16种**不同品种的花的照片，本项目使用的数据集包含3种不同的花(astilbe, bellflower, calendula)，每种有～1000张图片。

## Part 1: MLP Autoencoder 

构建一个MLP Autoencoder， 在数据集上进行训练，以得到一个可以对flower图片进行压缩编码与重建的autoencoder。

数据维度变化如下：3\*H\*W -> 256 -> 128 -> 64 (Latent vector) -> 128 -> 256 -> 3\*H\*W；隐藏层使用ReLU作为激活函数。

1. 运行`python MLPAE_train_script.py`可对MLP进行训练。相应的最优模型将被保存至`./models/BestModel_MLPAE.pkl`。

2. 可以通过运行`python visualization.py --model MLPAE`来可视化模型的输出效果。该脚本将生成的图片保存在`./vis/train_MLPAE.png`和`./vis/valid_MLPAE.png`中。你将看到原始图片与模型重建的图片之间的对比，以此来可视化模型的重建效果。


## Part 2: Convolutional Autoencoder 

观察到，MLP Autoencoder在flower数据集上的图像重建效果并不理想。这主要是因为MLP Autoencoder没有考虑到图片的空间结构，而将其当作一串向量来进行编码。

卷积神经网络（CNN）能够更好地提取图片的深层特征，因为它具有平移不变性（translation invariance）和局部性（locality）。本部分利用Pytorch搭建一个Convolutional Autoencoder模型，并在flower数据集上进行训练和评估。

数据维度变化如下：3 * H * W -> 32 * H/2 * W/2 -> 64 * H/4 * W/4 -> 128 * H/4 * W/4 -> encoding_dim -> 128 * H/4 * W/4 -> 64 * H/4 * W/4 -> 32 * H/2 * W/2 -> 3 * H * W；

使用Linear layer实现图片到latent vector的转换；

采用Max Pooling；采用ReLU作为激活函数；

1. 可以运行`python train_script.py --model AE`来进行模型训练，最优模型都将会被保存在`./models/BestModel_AE.pth`中。

2. 可以运行`python visualization.py --model AE`来可视化模型的重建效果，输出图片保存在`./vis/train_AE.png`与`./vis/valid_AE.png`中。你将看到原始图片与模型重建的图片的对比，以此来判断模型的重建效果。


encoder部分将一个图片压缩编码成为一个latent vector，该vector可以看作是图片的深层特征；decoder部分将这个vector重建回图片。因此，我们可以用Autoencoder的decoder部分来进行图片的生成：通过随机采样一个latent vector，可以期待decoder部分将这个vector转换成一个有意义的图片。

3. 可以运行`python random_generation.py --model AE`来生成一些图片，该脚本将会在$N(0, 1)$中随机采样若干vector，并feed进刚刚训练好的decoder将其转换成图片，输出图片保存在`./vis/random_images_AE.png`中。


## Part 3: Variational Autoencoder
part 2尝试通过在标准正态分布中采样向量，并将其输入到训练好的decoder中，从而生成了一些图片。

但是，观察可以发现，这些图片并不是特别“有意义”：它们看起来只是不同颜色像素的随机组合，而不是我们从未见过的“花”。这是因为传统的AE并不会将数据点映射到一个有意义的分布上，而是映射到一个个离散的点上。因此，当你随机采样时，采样到“有意义”的点的概率非常低。

本部分将搭建一个Variational Autoencoder (VAE)，以解决这一问题。

1. 可以运行`python train_script.py --model VAE`来进行模型训练。最优模型将被保存在`./models/BestModel_VAE.pth`中。

2. 可以运行`python visualization.py --model VAE`来可视化模型的重建效果，输出图片保存在`./vis/train_VAE.png`与`./vis/valid_VAE.png`中。

3. 可以运行`python random_generation.py --model VAE`来生成随机一些图片，该脚本将会在$N(0, 1)$中随机采样若干vector，并通过你训练好的decoder将其转换成图片，输出图片保存在`./vis/random_images_VAE.png`中。

VAE生成图片比Convolutional Autoencoder生成的图片更加“有意义”：这些图片看起来更像是真实的花，而不是像素的随机组合，虽然它们仍旧有些模糊。

## Part 4: Diffusion Model (DDPM) 

`Diffusion_image.py`中搭建了一个DDPM（Denoising Diffusion Probabilistic Models）算法框架。





**Reference**

[DDPM原论文](https://arxiv.org/pdf/2006.11239)
[算法推导，代码参考](https://www.bilibili.com/video/BV1b541197HX/?spm_id_from=333.788&vd_source=295aeb7cc6407338dd3e15d41a6b90ed)




