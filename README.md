# GAN-Introdution 更新中
### 1. 介绍GAN

- GAN的基本思想
- 为什么生成器不自己学？
- 为什么判别器不自己做?
- 具体算法 
- 笔记：
  + [李宏毅学习笔记30.GAN.01.](https://blog.csdn.net/oldmao_2001/article/details/105887797)
  + [李宏毅GAN教程（1）](https://zhuanlan.zhihu.com/p/57174645)

### 2. Gan的数学原理（GAN背后的理论）

- 笔记：
  + [李宏毅GAN教程（2）](https://zhuanlan.zhihu.com/p/57184819)
  + [李宏毅学习笔记33.GAN.04.Theory behind GAN](https://blog.csdn.net/oldmao_2001/article/details/105918115)

### 3. Conditional GAN (条件GAN)

- 笔记：
  + [李宏毅GAN教程（5）](https://zhuanlan.zhihu.com/p/57308383)
  + [李宏毅学习笔记31.GAN.02.Conditional Generation by GAN](https://blog.csdn.net/oldmao_2001/article/details/105903619)

# 1、 损失函数

| Level    | Title                                                        | Co-authors        | Publication                                    | Links                                                        |
| -------- | ------------------------------------------------------------ | ----------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| Beginner | LSGAN : Least Squares Generative Adversarial Networks        | Mao & et al.      | ICCV 2017                                      | [link](https://ieeexplore.ieee.org/document/8237566)         |
| Advanced | Improved Techniques for Training GANs                        | Salimans & et al. | NeurIPS (NIPS) 2016                            | [link](https://ceit.aut.ac.ir/http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf) |
| Advanced | WGAN : Wasserstein GAN                                       | Arjovsky & et al. | ICML 2017                                      | [link](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) |
| Advanced | WGAN-GP : improved Training of Wasserstein GANs              | 2017              | [link](https://arxiv.org/pdf/1704.00028v3.pdf) |                                                              |
| Advanced | Certifying Some Distributional Robustness with Principled Adversarial Training | Sinha & et al.    | ICML 2018                                      | [link](https://arxiv.org/pdf/1710.10571.pdf)[code](https://github.com/duchi-lab/certifiable-distributional-robustness) |

# 2、模型结构

| Level    | Title                                                        | Co-authors          | Publication                  | Links                                                        |
| -------- | ------------------------------------------------------------ | ------------------- | ---------------------------- | ------------------------------------------------------------ |
| Beginner | GAN : Generative Adversarial Nets                            | Goodfellow & et al. | NeurIPS (NIPS) 2014          | [link](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) |
| Beginner | GAN : Generative Adversarial Nets (Tutorial)                 | Goodfellow & et al. | NeurIPS (NIPS) 2016 Tutorial | [link](https://arxiv.org/pdf/1701.00160.pdf)                 |
| Beginner | CGAN : Conditional Generative Adversarial Nets               | Mirza & et al.      | -- 2014                      | [link](https://gist.github.com/shagunsodhani/5d726334de3014defeeb701099a3b4b3) |
| Beginner | InfoGAN : Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets | Chen & et al.       | NeuroIPS (NIPS) 2016         |                                                              |


# 3、实现方法

| Title                                                        | Co-authors                 | Publication       | Links                                                        | size            | FID/IS      |
| ------------------------------------------------------------ | -------------------------- | ----------------- | ------------------------------------------------------------ | --------------- | ----------- |
| Keras Implementation of GANs                                 | Linder-Norén               | Github            | [link](https://github.com/eriklindernoren/Keras-GAN)         |                 |             |
| GAN implementation hacks                                     | Salimans paper & Chintala  | World research    | [link](https://github.com/soumith/ganhacks)[paper](https://ceit.aut.ac.ir/~khalooei/tutorials/gan/#gan-hack-paper-2016) |                 |             |
| DCGAN : Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks | Radford & et al.           | 2015.11-ICLR 2016 | [link](https://github.com/carpedm20/DCGAN-tensorflow)[paper](https://arxiv.org/pdf/1511.06434.pdf) | 64x64 human     |             |
| ProGAN:Progressive Growing of GANs for Improved Quality, Stability, and Variation | Tero Karras                | 2017.10           | [paper](https://arxiv.org/pdf/1710.10196.pdf)[link](https://github.com/tkarras/progressive_growing_of_gans) | 1024x1024 human | 8.04        |
| SAGAN：Self-Attention Generative Adversarial Networks        | Han Zhang & Ian Goodfellow | 2018.05           | [paper](https://arxiv.org/pdf/1805.08318.pdf)[link](https://github.com/taki0112/Self-Attention-GAN-Tensorflow) | 128x128 obj     | 18.65/52.52 |
| BigGAN:Large Scale GAN Training for High Fidelity Natural Image Synthesis | Brock et al.               | 2018.09-ICLR 2019 | [demo](https://tfhub.dev/deepmind/biggan-256)[paper](https://arxiv.org/pdf/1809.11096.pdf)[link](https://github.com/AaronLeong/BigGAN-pytorch) | 512x512 obj     | 9.6/166.3   |
| StyleGAN:A Style-Based Generator Architecture for Generative Adversarial Networks | Tero Karras                | 2018.12           | [paper](https://arxiv.org/pdf/1812.04948.pdf)[link](https://github.com/NVlabs/stylegan) | 1024x1024 human | 4.04        |

## 3-1 GANs Applications in CV

| [图像翻译 (Image Translation)](https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/Image-translation图像翻译) | [超分辨率 (Super-Resolution)](https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/Super-Resolution超分辨率) | [图像上色(Colourful Image Colorization)](https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/Colourful-Image Colorization图像上色  ) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [图像修复(Image Inpainting)](https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/blob/master/GAN对抗生成网络/Image Inpainting图像修复/README.md) | [图像去噪(Image denoising)](https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/Image-denoising图像去噪) | [交互式图像生成](https://github.com/weslynn/AlphaTree-graphic-deep-neural-network/tree/master/GAN对抗生成网络/交互式图像生成) |






