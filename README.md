# PyTorch DDPM Implementation

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation, based on the paper ["Denoising Diffusion Probabilistic Models" by Ho et al.](https://arxiv.org/pdf/2006.11239)

## Overview

This project implements a complete DDPM pipeline that generates high-quality images by learning to reverse a diffusion process. The model progressively denoises random Gaussian noise into realistic images.

## Key Features

- **Custom UNet Architecture** with self-attention mechanisms
- **Diffusion Process** with 1000 timesteps and learned noise schedules
- **Time Embedding** using sinusoidal positional encoding
- **Multi-head Self-Attention** for enhanced feature learning
- **TensorBoard Integration** for training monitoring

## Technical Implementation

- **Framework**: PyTorch
- **Architecture**: UNet with skip connections and attention blocks
- **Training**: MSE loss between predicted and actual noise
- **Resolution**: 64x64 RGB images
- **Optimization**: AdamW optimizer with 3e-4 learning rate

## How It Works

### UNet Architecture
- **DoubleConv**: Consecutive convolution layers with GroupNorm and GELU activation
- **Down/Up Blocks**: Encoder-decoder structure with skip connections and time embedding
- **Self-Attention**: Multi-head attention for spatial feature refinement

### Diffusion Process
- **Forward**: Gradually adds Gaussian noise over 1000 timesteps
- **Reverse**: UNet learns to predict and remove noise at each step
- **Sampling**: Generates images by iteratively denoising random noise

### Training Algorithm
1. Sample images and random timesteps from dataset
2. Add noise according to timestep schedule
3. UNet predicts the added noise
4. Minimize MSE loss between predicted and actual noise
5. Repeat until convergence

## Results

- Model generates new images after each training epoch
- Training progress tracked via TensorBoard metrics
- Sample outputs saved as PNG files for evaluation

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) - Ho et al.