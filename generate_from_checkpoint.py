"""
Generate images from saved DDPM checkpoints
Usage: python generate_from_checkpoint.py <checkpoint_path> [num_images] [output_dir]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import logging
import argparse
from datetime import datetime
import math
import configparser

from torchvision.utils import make_grid


def load_config(config_path='config.ini'):
    """Load configuration from INI file"""
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Convert string values to appropriate types
    def parse_list(value, dtype=float):
        return [dtype(x.strip()) for x in value.split(',')]
    
    def parse_bool(value):
        return value.lower() in ('true', '1', 'yes', 'on')
    
    # Parse configuration
    cfg = {
        # Dataset
        'dataset': config.get('Dataset', 'dataset'),
        'img_size': tuple(parse_list(config.get('Dataset', 'img_size'), int)),
        
        # Model
        'timestep_embedding_dim': config.getint('Model', 'timestep_embedding_dim'),
        'n_layers': config.getint('Model', 'n_layers'),
        'hidden_dim': config.getint('Model', 'hidden_dim'),
        'n_timesteps': config.getint('Model', 'n_timesteps'),
        'beta_minmax': parse_list(config.get('Model', 'beta_minmax'), float),
        
        # Device
        'device': config.get('Device', 'device'),
        'gpu_id': config.getint('Device', 'gpu_id'),
        
        # Generation
        'checkpoint_generation_count': config.getint('Generation', 'checkpoint_generation_count'),
        'save_generated_individuals': parse_bool(config.get('Generation', 'save_generated_individuals')),
        'save_generated_grid': parse_bool(config.get('Generation', 'save_generated_grid'))
    }
    
    return cfg


# ============================================================================
# Model Classes (copied from main script)
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvBlock(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True, gn=False, gn_groups=8):

        if padding == 'same':
            padding = kernel_size // 2 * dilation

        super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation,
                                        groups=groups, bias=bias)

        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

    def forward(self, x, time_embedding=None, residual=False):
        if residual:
            x = x + time_embedding
            y = x
            x = super(ConvBlock, self).forward(x)
            y = y + x
        else:
            y = super(ConvBlock, self).forward(x)
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        return y


class Denoiser(nn.Module):
    def __init__(self, image_resolution, hidden_dims=[256, 256], diffusion_time_embedding_dim=256, n_times=1000):
        super(Denoiser, self).__init__()

        _, _, img_C = image_resolution

        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)

        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)

        self.time_project = nn.Sequential(
            ConvBlock(diffusion_time_embedding_dim, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=1))

        self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)])

        for idx in range(1, len(hidden_dims)):
            self.convs.append(ConvBlock(hidden_dims[idx - 1], hidden_dims[idx], kernel_size=3, dilation=3 ** ((idx - 1) // 2),
                                        activation_fn=True, gn=True, gn_groups=8))

        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)

    def forward(self, perturbed_x, diffusion_timestep):
        y = perturbed_x

        diffusion_embedding = self.time_embedding(diffusion_timestep)
        diffusion_embedding = self.time_project(diffusion_embedding.unsqueeze(-1).unsqueeze(-2))

        y = self.in_project(y)

        for i in range(len(self.convs)):
            y = self.convs[i](y, diffusion_embedding, residual=True)

        y = self.out_project(y)

        return y


class Diffusion(nn.Module):
    def __init__(self, model, image_resolution=[32, 32, 3], n_times=1000, beta_minmax=[1e-4, 2e-2], device='cuda'):
        super(Diffusion, self).__init__()

        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution

        self.model = model

        # define linear variance schedule(betas)
        beta_1, beta_T = beta_minmax
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device)
        self.sqrt_betas = torch.sqrt(betas)

        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.device = device

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def denoise_at_t(self, x_t, timestep, t):
        B, _, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)

        epsilon_pred = self.model(x_t, timestep)

        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)

        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred) + sqrt_beta * z

        return x_t_minus_1.clamp(-1., 1)

    def sample(self, N):
        x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)

        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            x_t = self.denoise_at_t(x_t, timestep, t)

        x_0 = self.reverse_scale_to_zero_to_one(x_t)
        return x_0


# ============================================================================
# Helper Functions
# ============================================================================

def save_single_image(x, idx, filename, output_dir):
    """Save a single image from a batch"""
    fig = plt.figure()
    plt.imshow(x[idx].transpose(0, 1).transpose(1, 2).detach().cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")


def save_sample_grid(x, postfix, filename, output_dir):
    """Save a grid of sample images"""
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def load_checkpoint(checkpoint_path, diffusion, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    timestamp = checkpoint.get('timestamp', 'Unknown')
    
    print(f"Loaded checkpoint from epoch {epoch} (loss: {loss:.6f}, saved: {timestamp})")
    return epoch, loss


def load_config_from_checkpoint_dir(checkpoint_path):
    """Load training configuration from the checkpoint directory"""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Try to find config files in the output directory
    output_dir = os.path.dirname(checkpoint_dir)
    config_paths = [
        os.path.join(output_dir, 'training_config_used.json'),
        os.path.join(output_dir, 'training_config.json'),
        os.path.join(checkpoint_dir, 'training_config_used.json'),
        os.path.join(checkpoint_dir, 'training_config.json')
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded training config from: {config_path}")
            return config
    
    print(f"Warning: No config file found in expected locations")
    return None


def generate_from_checkpoint(checkpoint_path, num_images=None, output_dir=None, config_path='config.ini'):
    """
    Generate images from a saved checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        num_images: Number of images to generate (uses config default if None)
        output_dir: Directory to save generated images
        config_path: Path to config file
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load configuration
    cfg = load_config(config_path)
    
    # Use config default if num_images not specified
    if num_images is None:
        num_images = cfg['checkpoint_generation_count']
    
    # Load training configuration from checkpoint directory
    training_config = load_config_from_checkpoint_dir(checkpoint_path)
    
    if training_config is None:
        print("Error: Could not load training configuration. Using config.ini values.")
        # Use config values as fallback
        img_size = cfg['img_size']
        hidden_dims = [cfg['hidden_dim'] for _ in range(cfg['n_layers'])]
        timestep_embedding_dim = cfg['timestep_embedding_dim']
        n_timesteps = cfg['n_timesteps']
        beta_minmax = cfg['beta_minmax']
        device_str = cfg['device']
    else:
        # Extract configuration from checkpoint
        img_size = tuple(training_config['img_size'])
        hidden_dims = [training_config['hidden_dim'] for _ in range(training_config['n_layers'])]
        timestep_embedding_dim = training_config['timestep_embedding_dim']
        n_timesteps = training_config['n_timesteps']
        beta_minmax = training_config['beta_minmax']
        device_str = training_config['device']
    
    # Setup device
    if device_str == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_str == 'cuda':
        device = torch.device(f"cuda:{cfg['gpu_id']}")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Initialize model with same architecture
    model = Denoiser(image_resolution=img_size,
                     hidden_dims=hidden_dims,
                     diffusion_time_embedding_dim=timestep_embedding_dim,
                     n_times=n_timesteps).to(device)
    
    diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps,
                          beta_minmax=beta_minmax, device=device).to(device)
    
    # Load checkpoint
    epoch, loss = load_checkpoint(checkpoint_path, diffusion, device=device)
    
    # Set output directory
    if output_dir is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        output_dir = os.path.join(checkpoint_dir, f'generated_from_epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_images} images...")
    model.eval()
    
    with torch.no_grad():
        generated_images = diffusion.sample(N=num_images)
    
    # Save individual generated images (if enabled in config)
    if cfg['save_generated_individuals']:
        for i in range(min(num_images, 10)):  # Save first 10 as individual images
            save_single_image(generated_images, idx=i, 
                             filename=f"generated_from_epoch_{epoch}_image_{i:02d}.png",
                             output_dir=output_dir)
    
    # Save grid (if enabled in config)
    if cfg['save_generated_grid']:
        save_sample_grid(generated_images, f"Generated from Epoch {epoch}", 
                        f"generated_grid_epoch_{epoch}.png", output_dir)
    
    print(f"Generated images saved to: {output_dir}")
    return generated_images


def main():
    parser = argparse.ArgumentParser(description='Generate images from DDPM checkpoint')
    parser.add_argument('checkpoint_path', help='Path to the checkpoint file')
    parser.add_argument('--num_images', type=int, help='Number of images to generate (uses config default if not specified)')
    parser.add_argument('--output_dir', help='Output directory for generated images')
    parser.add_argument('--config', default='config.ini', help='Path to config file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    print(f"Starting image generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    generate_from_checkpoint(args.checkpoint_path, args.num_images, args.output_dir, args.config)
    print(f"Image generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
