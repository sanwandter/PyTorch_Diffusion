# PyTorch DDPM Implementation

A comprehensive PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation, featuring both unconditional and conditional variants with classifier-free guidance.

## Overview

This project implements a complete DDPM pipeline that generates high-quality 64×64 images by learning to reverse a diffusion process. The model progressively denoises random Gaussian noise into realistic images, with support for class-conditional generation.

## Key Features

- **Unconditional DDPM**: Generate diverse images from pure noise
- **Conditional DDPM**: Class-conditional generation with classifier-free guidance  
- **Custom UNet Architecture** with multi-head self-attention mechanisms
- **Exponential Moving Average (EMA)** for improved model stability
- **Comprehensive Testing Suite** with flexible conditional/unconditional support

## Project Structure

```
├── ddpm.py                 # Unconditional DDPM implementation
├── ddpm_conditional.py     # Conditional DDPM with classifier-free guidance
├── modules.py              # UNet architectures and helper classes
├── model_test.py          # Testing script for both model types
└── utils.py               # Data loading and utility functions
```

## Diffusion Process

- **Forward**: Gradually adds Gaussian noise over 1000 timesteps
- **Reverse**: UNet learns to predict and remove noise at each step  
- **Sampling**: Generates images by iteratively denoising random noise

## Mathematical Foundation

### Forward Diffusion Process
Gradually adds noise over T=1000 timesteps:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

Where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod\_{s=1}^t \alpha_s$  
And $\beta_t$ increases linearly from $10^{-4}$ to $0.02$


### Training Objective
The model learns to predict noise $\epsilon$ at each timestep:

$$L = \mathbb{E}_{x_0, \epsilon, t} \left[ \|\epsilon - \epsilon\_\theta(x_t, t)\|^2 \right]$$

For conditional models with class labels $y$:

$$L = \mathbb{E}_{x_0, y, \epsilon, t} \left[ \|\epsilon - \epsilon\_\theta(x_t, t, y)\|^2 \right]$$

### Classifier-Free Guidance
Enhances conditional generation quality:

$$\tilde{\epsilon}\_\theta(x_t, t, y) = \epsilon\_\theta(x_t, t, \emptyset) + w \cdot (\epsilon\_\theta(x_t, t, y) - \epsilon\_\theta(x_t, t, \emptyset))$$

Where $w = 3$ is the guidance scale.

## Architecture Details

### UNet Components
- **DoubleConv**: Conv2d → GroupNorm → GELU → Conv2d → GroupNorm
- **Down Blocks**: MaxPool → DoubleConv → Time Embedding Integration
- **Up Blocks**: Upsample → Skip Connection → DoubleConv → Time Embedding
- **Self-Attention**: 4-head attention at resolutions 32×32, 16×16, 8×8

## Sampling Process

1. **Initialize**: Start with pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$
2. **Iterative Denoising**: For $t = T$ down to $1$:
   - Predict noise: $\epsilon_\theta(x_t, t)$ or $\epsilon_\theta(x_t, t, y)$
   - Apply classifier-free guidance (conditional only)
   - Denoise: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\_\theta(x_t, t)\right) + \sigma_t z$

## Results

- **Unconditional**: Diverse landscape image generation from pure noise
- **Conditional**: Class-specific CIFAR-10 image synthesis with enhanced quality via classifier-free guidance  
- **Resolution**: High-fidelity 64×64 RGB images
- **Performance**: Stable training with EMA and comprehensive monitoring via TensorBoard

## Technical Highlights

- **Custom UNet Implementation** with encoder-decoder architecture and skip connections
- **Multi-Head Self-Attention** integrated at multiple resolutions for enhanced spatial modeling
- **Classifier-Free Guidance** for improved conditional generation quality
- **Sinusoidal Time Embeddings** for effective timestep conditioning
- **Exponential Moving Averages** for training stability and better inference

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) - Ho & Salimans
