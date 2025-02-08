# MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders
[![License](https://img.shields.io/github/license/stanfordmimi/medvae?style=for-the-badge)](LICENSE)&nbsp;&nbsp;&nbsp;&nbsp;[![pypi](https://img.shields.io/pypi/v/medvae?style=for-the-badge)](https://pypi.org/project/medvae/)&nbsp;&nbsp;&nbsp;&nbsp;[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/stanfordmimi/MedVAE)

This repository contains the official PyTorch implementation for MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders.

![Overview](documentation/assets/overview.png)

## 🫁 What is MedVAE?

MedVAE is a family of six large-scale, generalizable 2D and 3D variational autoencoders (VAEs) designed for medical imaging. It is trained on over one million medical images across multiple anatomical regions and modalities. MedVAE autoencoders encode medical images as downsized latent representations and decode latent representations back to high-resolution images. Across diverse tasks obtained from 20 medical image datasets, we demonstrate that utilizing MedVAE latent representations in place of high-resolution images when training downstream models can lead to efficiency benefits (up to 70x improvement in throughput) while simultaneously preserving clinically-relevant features.

## ⚡️ Installation

To install MedVAE, you can simply run:

```python
pip install medvae
```

For an editable installation, use the following commands to clone and install this repository.
```python
git clone https://github.com/StanfordMIMI/MedVAE.git
cd medvae
pip install -e .[dev]
```

## 🚀 Usage Instructions

```python
import torch
from medvae import MVAE

fpath = "documentation/data/mmg_data/isJV8hQ2hhJsvEP5rdQNiy.png"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MVAE(model_name='medvae_4_3_2d', modality='xray').to(device)
img = model.apply_transform(fpath).to(device)

model.requires_grad_(False)
model.eval()

with torch.no_grad():
  latent = model(img)

```

We also developed an easy-to-use CLI inference tool for compressing your high-dimensional medical images into usable latents:

```python
medvae_inference -i INPUT_FOLDER -o OUTPUT_FOLDER -model_name MED_VAE_MODEL -modality MODALITY
```

For more information, please check our [inference documentation](/documentation/inference.md) and [demo](documentation/demo.ipynb).

## 📎 Citation
If you find this repository useful for your work, please cite the following paper:

```bibtex
@article{varma2025medvae,
  title = {MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders},
  author = {Maya Varma, Ashwin Kumar, Rogier van der Sluijs, Sophie Ostmeier, Louis Blankemeier, Pierre Chambon, Christian Bluethgen, Jip Prince, Curtis Langlotz, Akshay Chaudhari},
  year = {2025},
  publisher = {Github},
  journal = {Github},
  howpublished = {https://github.com/StanfordMIMI/MedVAE}
}
```

This repository is powered by [Hydra](https://github.com/facebookresearch/hydra) and [HuggingFace Accelerate](https://github.com/huggingface/accelerate). Our implementation of MedVAE is inspired by prior work on diffusion models from [CompVis](https://github.com/CompVis/latent-diffusion) and [Stability AI](https://github.com/Stability-AI/stablediffusion).
