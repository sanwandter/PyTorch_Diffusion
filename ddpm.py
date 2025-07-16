import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
from utils import get_data, setup_logging, save_images, plot_images
from modules import UNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Define the noise schedule (betas)
        self.betas = self.prepare_noise_schedule().to(device)

        # Pre-calculate alpha values for the forward diffusion process
        self.alpha = 1.0 - self.betas # α_t = 1 - β_t
        # Pre-calculate the cumulative product of alphas (alpha_hat)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # ᾱ_t = ∏(α_s) for s=1 to t
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        """
        Applies noise to the images `x` for a given timestep `t`.
        This is the forward process q(x_t | x_0).
        Applies noise to clean images according to the DDPM paper equation:
        x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε, where ε ~ N(0,I)

        Args:
            x: Clean images [batch_size, channels, height, width]
            t: Timesteps [batch_size]
        Returns:
            noisy_images: x_t 
            noise: ε (target for UNet to predict)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x) # Sample noise from a standard normal distribution
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
    
    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        """
        Reverse diffusion process: Generate images from noise
        Implements Algorithm 2 from DDPM paper "Sampling"
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # Start from pure noise x_T ~ N(0,I)
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)
            # Iteratively denoise for T steps
            for i in tqdm(reversed(range(self.noise_steps)), position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                pred_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # Reverse diffusion step - equation from DDPM paper
                x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * pred_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2  # Rescale to [0, 1]
        x = (x * 255).to(torch.uint8)  # Convert to uint8 for image representation
        return x
    

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join('runs', args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f'Starting epoch {epoch}:')
        pbar = tqdm(dataloader)
        # Algorithm 1 from the DDPM paper
        # "Training"
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            # Sample random timesteps
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # Add noise to the images
            x_t, noise = diffusion.noise_images(images, t)
            # Predict the noise using the model
            pred_noise = model(x_t, t) # ε_θ(x_t, t)
            # Calculate the loss
            loss = mse(pred_noise, noise) # ||ε - ε_θ(x_t, t)||²

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar('MSE', loss.item(), global_step=epoch * l + i)
        
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join('results', args.run_name, f'{epoch}.png'))
        torch.save(model.state_dict(), os.path.join('models', args.run_name, 'last.pt'))


# Training script entry point
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = 'ddpm_run'
    args.epochs = 500
    args.batch_size = 12
    args.img_size = 64
    args.dataset_path = '../dataset/landscapes'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()