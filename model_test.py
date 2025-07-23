import torch
from ddpm import Diffusion
from modules import UNet
import os
from utils import save_images
import argparse

def generate_images(args):
    # Set up device
    device = args.device

    # Initialize Diffusion and UNet model
    diffusion = Diffusion(img_size=args.img_size, device=device)
    model = UNet().to(device)

    # Load the trained model
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print(f"Model loaded successfully from {args.model_path}")
    else:
        print(f"Error: Model not found at {args.model_path}")
        return

    # Generate images
    num_images_to_generate = 12
    print(f"Generating {num_images_to_generate} images...")
    generated_images = diffusion.sample(model, num_images_to_generate)

    # Save the generated images
    output_dir = 'results/generated_images'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.model}_images{args.idx}.png')
    save_images(generated_images, output_path, nrow=4)

if __name__ == '__main__':  
    for i in range(10):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.model_path = 'models/ddpm_run/landscape.pt' # Change this to model path
        args.img_size = 64 
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        args.model = 'landscape' # Change this to model name
        args.idx = i
        generate_images(args)




