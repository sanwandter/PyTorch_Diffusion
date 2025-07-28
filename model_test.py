import torch
from ddpm import Diffusion
from ddpm_conditional import Diffusion as ConditionalDiffusion
from modules import UNet, UNetConditional
import os
from utils import save_images
import argparse

def generate_images(args):
    # Set up device
    device = args.device

    if args.conditional:
        # Conditional model setup
        diffusion = ConditionalDiffusion(img_size=args.img_size, device=device)
        model = UNetConditional(num_classes=args.num_classes).to(device)
        
        # Load the trained conditional model
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path))
            print(f"Conditional model loaded successfully from {args.model_path}")
        else:
            print(f"Error: Model not found at {args.model_path}")
            return

        # Generate 10 images for one specific class based on idx
        class_label = args.idx % args.num_classes  # Use idx to cycle through classes
        print(f"Generating 10 images for class {class_label}...")
        labels = torch.full((10,), class_label).long().to(device)
        generated_images = diffusion.sample(model, n=10, labels=labels)
        
        # Save images for this class (2 rows of 5 images each)
        output_dir = 'results/generated_images'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{args.model}_class{class_label}.png')
        save_images(generated_images, output_path, nrow=5)
        print(f"Generated images for class {class_label} saved to {output_path}")
    else:
        # Original unconditional model setup
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
        args.model_path = 'models/ddpm_conditional_run/cifar10.pt' # Change this to model path
        args.model = 'cifar10' # Change this to model name (landscape, cifar10, etc.)

        # Configuration flags
        args.conditional = True  # Set to True for conditional model, False for unconditional
        
        if args.conditional:
            # Conditional model settings
            args.num_classes = 10
        else:
            # Unconditional model settings  
            args.num_classes = None
            
        args.img_size = 64 
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        args.idx = i
        generate_images(args)




