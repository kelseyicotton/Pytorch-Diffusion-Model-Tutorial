"""
Denoising Diffusion Probabilistic Model

Kernel Author: Minsu Kang
Email: mskang1478@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime
import configparser
import argparse

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
from torch.utils.tensorboard import SummaryWriter


def load_config(config_path='config.ini'):
    """Load configuration from INI file"""
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Convert string values to appropriate types
    def parse_list(value, dtype=float):
        return [dtype(x.strip()) for x in value.split(',')]
    
    def parse_bool(value):
        return value.lower() in ('true', '1', 'yes', 'on')
    
    # Helper function to get config values with defaults
    def get_config(section, key, default=None, dtype=str):
        try:
            if dtype == bool:
                return parse_bool(config.get(section, key))
            elif dtype == int:
                return config.getint(section, key)
            elif dtype == float:
                return config.getfloat(section, key)
            else:
                return config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            if default is not None:
                return default
            else:
                raise
    
    # Parse configuration with defaults
    cfg = {
        # Dataset
        'dataset': get_config('Dataset', 'dataset', 'MNIST'),
        'dataset_path': get_config('Dataset', 'dataset_path', '~/datasets'),
        'img_size': tuple(parse_list(get_config('Dataset', 'img_size', '28, 28, 1'), int)),
        'num_workers': get_config('Dataset', 'num_workers', 1, int),
        'pin_memory': get_config('Dataset', 'pin_memory', True, bool),
        
        # Model
        'timestep_embedding_dim': get_config('Model', 'timestep_embedding_dim', 256, int),
        'n_layers': get_config('Model', 'n_layers', 8, int),
        'hidden_dim': get_config('Model', 'hidden_dim', 256, int),
        'n_timesteps': get_config('Model', 'n_timesteps', 1000, int),
        'beta_minmax': parse_list(get_config('Model', 'beta_minmax', '1e-4, 2e-2'), float),
        
        # Training
        'train_batch_size': get_config('Training', 'train_batch_size', 128, int),
        'inference_batch_size': get_config('Training', 'inference_batch_size', 64, int),
        'lr': get_config('Training', 'lr', 5e-5, float),
        'epochs': get_config('Training', 'epochs', 100, int),
        'save_every_n_epochs': get_config('Training', 'save_every_n_epochs', 10, int),
        'seed': get_config('Training', 'seed', 1234, int),
        
        # Visualization
        'denoising_viz_every': get_config('Visualization', 'denoising_viz_every', 5, int),
        'checkpoint_samples_every': get_config('Visualization', 'checkpoint_samples_every', 10, int),
        'checkpoint_num_samples': get_config('Visualization', 'checkpoint_num_samples', 16, int),
        'loss_analysis_every': get_config('Visualization', 'loss_analysis_every', 10, int),
        'param_hist_every': get_config('Visualization', 'param_hist_every', 10, int),
        
        # Output
        'base_output_dir': get_config('Output', 'base_output_dir', 'outputs'),
        'save_individual_images': get_config('Output', 'save_individual_images', True, bool),
        'save_grid_images': get_config('Output', 'save_grid_images', True, bool),
        'image_dpi': get_config('Output', 'image_dpi', 150, int),
        
        # Device
        'device': get_config('Device', 'device', 'auto'),
        'gpu_id': get_config('Device', 'gpu_id', 0, int),
        
        # Logging
        'log_level': get_config('Logging', 'log_level', 'INFO'),
        'save_logs': get_config('Logging', 'save_logs', True, bool),
        'save_architecture': get_config('Logging', 'save_architecture', True, bool),
        
        # TensorBoard
        'enable_tensorboard': get_config('TensorBoard', 'enable_tensorboard', True, bool),
        'log_images': get_config('TensorBoard', 'log_images', True, bool),
        'log_parameters': get_config('TensorBoard', 'log_parameters', True, bool),
        
        # Generation
        'final_generation_count': get_config('Generation', 'final_generation_count', 64, int),
        'checkpoint_generation_count': get_config('Generation', 'checkpoint_generation_count', 20, int),
        'save_generated_individuals': get_config('Generation', 'save_generated_individuals', True, bool),
        'save_generated_grid': get_config('Generation', 'save_generated_grid', True, bool)
    }
    
    return cfg


# Default configuration (fallback)
default_config = {
    'dataset': 'MNIST',
    'dataset_path': '~/datasets',
    'img_size': (28, 28, 1),
    'num_workers': 1,
    'pin_memory': True,
    'timestep_embedding_dim': 256,
    'n_layers': 8,
    'hidden_dim': 256,
    'n_timesteps': 1000,
    'beta_minmax': [1e-4, 2e-2],
    'train_batch_size': 128,
    'inference_batch_size': 64,
    'lr': 5e-5,
    'epochs': 100,
    'save_every_n_epochs': 10,
    'seed': 1234,
    'denoising_viz_every': 5,
    'checkpoint_samples_every': 10,
    'checkpoint_num_samples': 16,
    'loss_analysis_every': 10,
    'param_hist_every': 10,
    'base_output_dir': 'outputs',
    'save_individual_images': True,
    'save_grid_images': True,
    'image_dpi': 150,
    'device': 'auto',
    'gpu_id': 0,
    'log_level': 'INFO',
    'save_logs': True,
    'save_architecture': True,
    'enable_tensorboard': True,
    'log_images': True,
    'log_parameters': True,
    'final_generation_count': 64,
    'checkpoint_generation_count': 20,
    'save_generated_individuals': True,
    'save_generated_grid': True
}

# Load configuration with fallback
try:
    config = load_config()
    print("Configuration loaded successfully")
except Exception as e:
    print(f"Error loading configuration: {e}")
    print("Using default configuration values...")
    config = default_config.copy()

# Ensure all required keys exist in config
for key, default_value in default_config.items():
    if key not in config:
        print(f"Warning: Missing config key '{key}', using default value: {default_value}")
        config[key] = default_value

# Debug: Print config keys to help troubleshoot
print(f"Config keys available: {list(config.keys())}")
print(f"log_images value: {config.get('log_images', 'NOT FOUND')}")
print(f"enable_tensorboard value: {config.get('enable_tensorboard', 'NOT FOUND')}")

# Add a check to detect if config is being modified
def check_config_integrity():
    """Check if config has been modified"""
    missing_keys = []
    for key in original_config.keys():
        if key not in config:
            missing_keys.append(key)
    if missing_keys:
        print(f"WARNING: Config keys were removed: {missing_keys}")
        return False
    return True

# Safety function to ensure config keys exist
def ensure_config_key(key, default_value):
    """Ensure a config key exists, add default if missing"""
    if key not in config:
        print(f"Warning: Missing config key '{key}', using default value: {default_value}")
        config[key] = default_value
    return config[key]

# Create a frozen copy of the config to prevent modifications
original_config = config.copy()
frozen_config = config.copy()

def get_config_value(key, default_value):
    """Get config value with fallback to original config and then default"""
    # First try current config
    if key in config:
        return config[key]
    # Then try original config
    elif key in original_config:
        print(f"Warning: Config key '{key}' was modified, restoring from original config")
        config[key] = original_config[key]
        return config[key]
    # Finally use default
    else:
        print(f"Warning: Config key '{key}' not found, using default value: {default_value}")
        print(f"Available keys: {list(config.keys())}")
        print(f"Original keys: {list(original_config.keys())}")
        config[key] = default_value
        return default_value

# Create timestamped output directory first
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f"{config['base_output_dir']}_{timestamp}"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, 'tensorboard')
SAMPLES_DIR = os.path.join(OUTPUT_DIR, 'checkpoint_samples')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
DENOISING_DIR = os.path.join(VISUALIZATIONS_DIR, 'denoising_process')
NOISE_COMPARISON_DIR = os.path.join(VISUALIZATIONS_DIR, 'noise_comparison')
FORWARD_PROCESS_DIR = os.path.join(VISUALIZATIONS_DIR, 'forward_process')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(DENOISING_DIR, exist_ok=True)
os.makedirs(NOISE_COMPARISON_DIR, exist_ok=True)
os.makedirs(FORWARD_PROCESS_DIR, exist_ok=True)

# Initialize TensorBoard writer (only if enabled)
writer = SummaryWriter(log_dir=TENSORBOARD_DIR) if config['enable_tensorboard'] else None

# Setup logging with timestamp (both console and file)
log_level = getattr(logging, config['log_level'].upper())
handlers = [logging.StreamHandler()]  # Console output
if config['save_logs']:
    handlers.append(logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log')))  # File output

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=handlers
)
logger = logging.getLogger(__name__)

logger.info(f"Created output directory: {OUTPUT_DIR}")
logger.info(f"Created checkpoint directory: {CHECKPOINT_DIR}")
if config['enable_tensorboard']:
    logger.info(f"Created TensorBoard directory: {TENSORBOARD_DIR}")
logger.info(f"Created checkpoint samples directory: {SAMPLES_DIR}")
logger.info(f"Created visualizations directory: {VISUALIZATIONS_DIR}")
logger.info(f"Created denoising process directory: {DENOISING_DIR}")
logger.info(f"Created noise comparison directory: {NOISE_COMPARISON_DIR}")
logger.info(f"Created forward process directory: {FORWARD_PROCESS_DIR}")


# ============================================================================
# Model Hyperparameters (from config)
# ============================================================================

dataset_path = config['dataset_path']

# Device setup
if config['device'] == 'auto':
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if cuda else "cpu")
elif config['device'] == 'cuda':
    DEVICE = torch.device(f"cuda:{config['gpu_id']}")
else:
    DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

dataset = config['dataset']
img_size = config['img_size']

timestep_embedding_dim = config['timestep_embedding_dim']
n_layers = config['n_layers']
hidden_dim = config['hidden_dim']
n_timesteps = config['n_timesteps']
beta_minmax = config['beta_minmax']

train_batch_size = config['train_batch_size']
inference_batch_size = config['inference_batch_size']
lr = config['lr']
epochs = config['epochs']
save_every_n_epochs = config['save_every_n_epochs']

seed = config['seed']

hidden_dims = [hidden_dim for _ in range(n_layers)]
torch.manual_seed(seed)
np.random.seed(seed)

# Log hyperparameters
logger.info("=" * 60)
logger.info("Model Hyperparameters:")
logger.info(f"  Dataset: {dataset}")
logger.info(f"  Image size: {img_size}")
logger.info(f"  Timestep embedding dim: {timestep_embedding_dim}")
logger.info(f"  Number of layers: {n_layers}")
logger.info(f"  Hidden dim: {hidden_dim}")
logger.info(f"  Number of timesteps: {n_timesteps}")
logger.info(f"  Beta min/max: {beta_minmax}")
logger.info(f"  Train batch size: {train_batch_size}")
logger.info(f"  Inference batch size: {inference_batch_size}")
logger.info(f"  Learning rate: {lr}")
logger.info(f"  Epochs: {epochs}")
logger.info(f"  Save every N epochs: {save_every_n_epochs}")
logger.info(f"  Seed: {seed}")
logger.info("=" * 60)

# Log hyperparameters to TensorBoard
hparams = {
    'dataset': dataset,
    'img_size': str(img_size),
    'timestep_embedding_dim': timestep_embedding_dim,
    'n_layers': n_layers,
    'hidden_dim': hidden_dim,
    'n_timesteps': n_timesteps,
    'beta_min': beta_minmax[0],
    'beta_max': beta_minmax[1],
    'train_batch_size': train_batch_size,
    'inference_batch_size': inference_batch_size,
    'lr': lr,
    'epochs': epochs,
    'save_every_n_epochs': save_every_n_epochs,
    'seed': seed,
    'device': str(DEVICE)
}

# Training configuration will be saved after model initialization


# ============================================================================
# Step 1. Load (or download) Dataset
# ============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])

kwargs = {'num_workers': config['num_workers'], 'pin_memory': config['pin_memory']}

if dataset == 'CIFAR10':
    train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
    test_dataset = CIFAR10(dataset_path, transform=transform, train=False, download=True)
else:
    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)

logger.info(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")


# ============================================================================
# Step 2. Define our model: Denoising Diffusion Probabilistic Models (DDPMs)
# ============================================================================

# Sinusoidal embedding for diffusion timestep
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


# In this tutorial, we use a simple stacked-convolution model with various dilations instead of UNet-like architecture.
class ConvBlock(nn.Conv2d):
    """
        Conv2D Block
            Args:
                x: (N, C_in, H, W)
            Returns:
                y: (N, C_out, H, W)
    """

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
            # in the paper, diffusion timestep embedding was only applied to residual blocks of U-Net
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


# Define Gaussian Diffusion
class Diffusion(nn.Module):
    def __init__(self, model, image_resolution=[32, 32, 3], n_times=1000, beta_minmax=[1e-4, 2e-2], device='cuda'):

        super(Diffusion, self).__init__()

        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution

        self.model = model

        # define linear variance schedule(betas)
        beta_1, beta_T = beta_minmax
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device)  # follows DDPM paper
        self.sqrt_betas = torch.sqrt(betas)

        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.device = device

    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def make_noisy(self, x_zeros, t):
        # perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)
        epsilon = torch.randn_like(x_zeros).to(self.device)

        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)

        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar

        return noisy_sample.detach(), epsilon

    def forward(self, x_zeros):
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)

        B, _, _, _ = x_zeros.shape

        # (1) randomly choose diffusion time-step
        t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(self.device)

        # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
        perturbed_images, epsilon = self.make_noisy(x_zeros, t)

        # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
        pred_epsilon = self.model(perturbed_images, t)

        return perturbed_images, epsilon, pred_epsilon

    def denoise_at_t(self, x_t, timestep, t):
        B, _, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)

        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred = self.model(x_t, timestep)

        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)

        # denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred) + sqrt_beta * z

        return x_t_minus_1.clamp(-1., 1)

    def sample(self, N):
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)

        # autoregressively denoise from x_T to x_0
        #     i.e., generate image from noise, x_T
        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            x_t = self.denoise_at_t(x_t, timestep, t)

        # denormalize x_0 into 0 ~ 1 ranged values.
        x_0 = self.reverse_scale_to_zero_to_one(x_t)

        return x_0


# ============================================================================
# Helper Functions for Saving Images
# ============================================================================

def save_single_image(x, idx, filename, output_dir=None):
    """Save a single image from a batch"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    fig = plt.figure()
    plt.imshow(x[idx].transpose(0, 1).transpose(1, 2).detach().cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {filename}")


def save_sample_grid(x, postfix, filename, output_dir=None):
    """Save a grid of sample images"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {filename}")


def save_visualization_grid(images, title, filename, output_dir, num_cols=4):
    """Save a grid of visualization images with custom layout"""
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        
        # Convert tensor to numpy and transpose for display
        img_np = img.detach().cpu().numpy()
        if img_np.shape[0] == 1:  # Grayscale
            ax.imshow(img_np[0], cmap='gray')
        else:  # RGB
            ax.imshow(np.transpose(img_np, (1, 2, 0)))
        
        ax.set_title(f"Step {i}")
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Saved visualization: {filename}")


def save_noise_comparison(actual_noise, predicted_noise, difference, epoch, output_dir):
    """Save noise comparison visualization"""
    # Create comparison grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Actual noise
    axes[0, 0].imshow(np.transpose(make_grid(actual_noise[:4].detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    axes[0, 0].set_title("Actual Noise")
    axes[0, 0].axis('off')
    
    # Predicted noise
    axes[0, 1].imshow(np.transpose(make_grid(predicted_noise[:4].detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    axes[0, 1].set_title("Predicted Noise")
    axes[0, 1].axis('off')
    
    # Difference
    axes[0, 2].imshow(np.transpose(make_grid(difference[:4].detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    axes[0, 2].set_title("Difference (Error)")
    axes[0, 2].axis('off')
    
    # Individual samples
    for i in range(3):
        axes[1, i].imshow(np.transpose(actual_noise[i].detach().cpu(), (1, 2, 0)))
        axes[1, i].set_title(f"Sample {i}")
        axes[1, i].axis('off')
    
    plt.suptitle(f"Noise Prediction Comparison - Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"noise_comparison_epoch_{epoch:03d}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Saved noise comparison: noise_comparison_epoch_{epoch:03d}.png")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_denoising_process(diffusion, writer, epoch):
    """Log denoising process visualization to TensorBoard and save images"""
    model = diffusion.model
    model.eval()
    
    with torch.no_grad():
        # Get a batch from test data for visualization
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(DEVICE)
            break
        
        # Take first 4 samples for visualization
        x_vis = x[:4]
        
        # 1. Forward process visualization (add noise at different timesteps)
        timesteps_to_show = [0, 100, 200, 500, 800, 999]
        forward_images = []
        
        for t in timesteps_to_show:
            t_tensor = torch.full((4,), t, device=DEVICE, dtype=torch.long)
            noisy_x, _ = diffusion.make_noisy(x_vis, t_tensor)
            noisy_x = diffusion.reverse_scale_to_zero_to_one(noisy_x)
            forward_images.append(noisy_x)
        
        # Log forward process to TensorBoard
        if writer is not None and get_config_value('log_images', True):
            forward_grid = torch.cat(forward_images, dim=0)
            writer.add_images(f"Denoising/Forward_Process_Epoch_{epoch}", forward_grid, epoch)
        
        # Save forward process images
        save_visualization_grid(forward_images, 
                               f"Forward Process - Epoch {epoch}", 
                               f"forward_process_epoch_{epoch:03d}.png", 
                               DENOISING_DIR)
        
        # 2. Noise prediction vs actual noise
        t_random = torch.randint(0, diffusion.n_times, (4,), device=DEVICE)
        noisy_x, actual_noise = diffusion.make_noisy(x_vis, t_random)
        predicted_noise = model(noisy_x, t_random)
        noise_difference = torch.abs(actual_noise - predicted_noise)
        
        # Log noise comparison to TensorBoard
        if writer is not None and get_config_value('log_images', True):
            writer.add_images(f"Denoising/Actual_Noise_Epoch_{epoch}", actual_noise, epoch)
            writer.add_images(f"Denoising/Predicted_Noise_Epoch_{epoch}", predicted_noise, epoch)
            writer.add_images(f"Denoising/Noise_Difference_Epoch_{epoch}", noise_difference, epoch)
        
        # Save noise comparison images
        save_noise_comparison(actual_noise, predicted_noise, noise_difference, epoch, NOISE_COMPARISON_DIR)
        
        # 3. Denoising steps visualization (reverse process)
        # Start from pure noise and show denoising steps
        x_t = torch.randn_like(x_vis)
        denoising_steps = []
        
        # Show denoising at key timesteps
        reverse_timesteps = [999, 800, 600, 400, 200, 0]
        
        for t in reverse_timesteps:
            if t > 0:
                t_tensor = torch.full((4,), t, device=DEVICE, dtype=torch.long)
                x_t = diffusion.denoise_at_t(x_t, t_tensor, t)
            
            # Convert to 0-1 range for visualization
            x_vis_step = diffusion.reverse_scale_to_zero_to_one(x_t)
            denoising_steps.append(x_vis_step)
        
        # Log reverse process to TensorBoard
        if writer is not None and get_config_value('log_images', True):
            reverse_grid = torch.cat(denoising_steps, dim=0)
            writer.add_images(f"Denoising/Reverse_Process_Epoch_{epoch}", reverse_grid, epoch)
        
        # Save reverse process images
        save_visualization_grid(denoising_steps, 
                               f"Reverse Process (Denoising) - Epoch {epoch}", 
                               f"reverse_process_epoch_{epoch:03d}.png", 
                               DENOISING_DIR)
        
        # 4. Timestep distribution
        if writer is not None:
            timestep_hist = torch.histogram(t_random.float(), bins=20, range=(0, diffusion.n_times))
            writer.add_histogram(f"Denoising/Timestep_Distribution_Epoch_{epoch}", t_random.float(), epoch)
        
        # Save timestep distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(t_random.cpu().numpy(), bins=20, alpha=0.7, edgecolor='black')
        plt.title(f"Timestep Distribution - Epoch {epoch}")
        plt.xlabel("Timestep")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(DENOISING_DIR, f"timestep_distribution_epoch_{epoch:03d}.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Saved timestep distribution: timestep_distribution_epoch_{epoch:03d}.png")
    
    model.train()  # Return to training mode


def generate_checkpoint_samples(diffusion, epoch, num_samples=16, samples_dir=SAMPLES_DIR):
    """Generate sample images at checkpoint and save them"""
    logger.info(f"Generating {num_samples} samples for epoch {epoch}...")
    
    # Create epoch-specific samples directory
    epoch_samples_dir = os.path.join(samples_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_samples_dir, exist_ok=True)
    
    # Generate samples
    model = diffusion.model
    model.eval()
    with torch.no_grad():
        generated_samples = diffusion.sample(N=num_samples)
    
    # Save individual samples
    for i in range(min(num_samples, 16)):  # Save up to 16 individual images
        save_single_image(generated_samples, idx=i, 
                         filename=f"sample_{i:02d}.png", 
                         output_dir=epoch_samples_dir)
    
    # Save sample grid
    save_sample_grid(generated_samples, f"Epoch {epoch} Samples", 
                    f"epoch_{epoch:03d}_grid.png", epoch_samples_dir)
    
    logger.info(f"Checkpoint samples saved to: {epoch_samples_dir}")
    return generated_samples


def save_checkpoint(diffusion, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': diffusion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # Save epoch-specific checkpoint
    epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, epoch_path)
    
    logger.info(f"Saved checkpoint: {epoch_path}")
    return latest_path, epoch_path


def load_checkpoint(checkpoint_path, diffusion, optimizer=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    timestamp = checkpoint.get('timestamp', 'Unknown')
    
    logger.info(f"Loaded checkpoint from epoch {epoch} (loss: {loss:.6f}, saved: {timestamp})")
    return epoch, loss


# ============================================================================
# Initialize Model
# ============================================================================

model = Denoiser(image_resolution=img_size,
                 hidden_dims=hidden_dims,
                 diffusion_time_embedding_dim=timestep_embedding_dim,
                 n_times=n_timesteps).to(DEVICE)

diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps,
                      beta_minmax=beta_minmax, device=DEVICE).to(DEVICE)

optimizer = Adam(diffusion.parameters(), lr=lr)
denoising_loss = nn.MSELoss()

logger.info(f"Number of model parameters: {count_parameters(diffusion)}")

# Log model architecture to TensorBoard
if writer is not None:
    writer.add_text("Model/Architecture", f"Total parameters: {count_parameters(diffusion):,}", 0)
    writer.add_text("Model/Architecture", f"Hidden dimensions: {hidden_dims}", 0)
    writer.add_text("Model/Architecture", f"Image resolution: {img_size}", 0)
    writer.add_text("Model/Architecture", f"Timestep embedding dim: {timestep_embedding_dim}", 0)
    writer.add_text("Model/Architecture", f"Number of timesteps: {n_timesteps}", 0)

    # Log model graph (if possible)
    try:
        # Create a dummy input for model graph visualization
        dummy_input = torch.randn(1, img_size[2], img_size[0], img_size[1]).to(DEVICE)
        dummy_timestep = torch.randint(0, n_timesteps, (1,)).to(DEVICE)
        writer.add_graph(model, (dummy_input, dummy_timestep))
        logger.info("Model graph logged to TensorBoard")
    except Exception as e:
        logger.warning(f"Could not log model graph to TensorBoard: {e}")

# ============================================================================
# Detailed Model Architecture Logging
# ============================================================================

# Create separate architecture logger
arch_logger = logging.getLogger('architecture')
arch_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, 'model_architecture.log'))
arch_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
arch_handler.setFormatter(arch_formatter)
arch_logger.addHandler(arch_handler)
arch_logger.setLevel(logging.INFO)

# Log architecture details to separate file
arch_logger.info("=" * 60)
arch_logger.info("MODEL ARCHITECTURE DETAILS")
arch_logger.info("=" * 60)

# Input dimensions
arch_logger.info(f"Input Image Dimensions:")
arch_logger.info(f"  - Batch size: {train_batch_size} (training), {inference_batch_size} (inference)")
arch_logger.info(f"  - Image size: {img_size[0]}×{img_size[1]}×{img_size[2]} (H×W×C)")
arch_logger.info(f"  - Total pixels per image: {img_size[0] * img_size[1] * img_size[2]:,}")

# Network architecture
arch_logger.info(f"\nDenoiser Network Architecture:")
arch_logger.info(f"  - Number of layers: {n_layers}")
arch_logger.info(f"  - Hidden dimension: {hidden_dim}")
arch_logger.info(f"  - Hidden dimensions per layer: {hidden_dims}")

# Time conditioning
arch_logger.info(f"\nTime Conditioning:")
arch_logger.info(f"  - Time embedding dimension: {timestep_embedding_dim}")
arch_logger.info(f"  - Number of timesteps: {n_timesteps}")
arch_logger.info(f"  - Beta schedule: {beta_minmax[0]:.2e} to {beta_minmax[1]:.2e}")

# Layer-by-layer breakdown
arch_logger.info(f"\nLayer-by-Layer Breakdown:")
arch_logger.info(f"  1. Input Projection: Conv7×7, {img_size[2]}→{hidden_dim} channels")
arch_logger.info(f"  2. Time Projection: Conv1×1×2, {timestep_embedding_dim}→{hidden_dim} channels")

for i in range(n_layers):
    dilation = 3 ** ((i-1) // 2) if i > 0 else 1
    receptive_field = 1 + 2 * dilation
    arch_logger.info(f"  {i+3}. Conv Block {i}: Conv3×3 (dilation={dilation}, RF≈{receptive_field}×{receptive_field}), {hidden_dim}→{hidden_dim} channels")

arch_logger.info(f"  {n_layers+3}. Output Projection: Conv3×3, {hidden_dim}→{img_size[2]} channels")

# Memory and computation estimates
total_params = count_parameters(diffusion)
arch_logger.info(f"\nModel Statistics:")
arch_logger.info(f"  - Total parameters: {total_params:,}")
arch_logger.info(f"  - Parameters per layer (avg): {total_params // n_layers:,}")
arch_logger.info(f"  - Model size (MB): {total_params * 4 / (1024**2):.2f}")

# Forward pass dimension flow
arch_logger.info(f"\nForward Pass Dimension Flow:")
arch_logger.info(f"  Input: [{train_batch_size}, {img_size[2]}, {img_size[0]}, {img_size[1]}]")
arch_logger.info(f"  ↓ Input Projection")
arch_logger.info(f"  Features: [{train_batch_size}, {hidden_dim}, {img_size[0]}, {img_size[1]}]")
arch_logger.info(f"  ↓ {n_layers} Conv Blocks with Time Conditioning")
arch_logger.info(f"  Features: [{train_batch_size}, {hidden_dim}, {img_size[0]}, {img_size[1]}]")
arch_logger.info(f"  ↓ Output Projection")
arch_logger.info(f"  Predicted Noise: [{train_batch_size}, {img_size[2]}, {img_size[0]}, {img_size[1]}]")

arch_logger.info("=" * 60)

# Also log to main logger for console output
logger.info("Model architecture details saved to: model_architecture.log")

# Save training configuration to file (now that model is initialized)
config = {
    'dataset': dataset,
    'img_size': img_size,
    'timestep_embedding_dim': timestep_embedding_dim,
    'n_layers': n_layers,
    'hidden_dim': hidden_dim,
    'n_timesteps': n_timesteps,
    'beta_minmax': beta_minmax,
    'train_batch_size': train_batch_size,
    'inference_batch_size': inference_batch_size,
    'lr': lr,
    'epochs': epochs,
    'save_every_n_epochs': save_every_n_epochs,
    'seed': seed,
    'device': str(DEVICE),
    'timestamp': timestamp,
    # Architecture details
    'architecture': {
        'input_dimensions': {
            'batch_size_training': train_batch_size,
            'batch_size_inference': inference_batch_size,
            'image_height': img_size[0],
            'image_width': img_size[1],
            'image_channels': img_size[2],
            'total_pixels_per_image': img_size[0] * img_size[1] * img_size[2]
        },
        'network_units': {
            'number_of_layers': n_layers,
            'hidden_dimension': hidden_dim,
            'hidden_dimensions_per_layer': hidden_dims,
            'total_parameters': total_params,
            'model_size_mb': total_params * 4 / (1024**2)
        },
        'conditioning_vector': {
            'time_embedding_dimension': timestep_embedding_dim,
            'number_of_timesteps': n_timesteps,
            'beta_schedule_min': beta_minmax[0],
            'beta_schedule_max': beta_minmax[1]
        },
        'layer_breakdown': [
            {
                'layer': 1,
                'type': 'Input Projection',
                'operation': f'Conv7×7, {img_size[2]}→{hidden_dim} channels'
            },
            {
                'layer': 2,
                'type': 'Time Projection',
                'operation': f'Conv1×1×2, {timestep_embedding_dim}→{hidden_dim} channels'
            }
        ] + [
            {
                'layer': i + 3,
                'type': f'Conv Block {i}',
                'operation': f'Conv3×3 (dilation={3 ** ((i-1) // 2) if i > 0 else 1}, RF≈{1 + 2 * (3 ** ((i-1) // 2) if i > 0 else 1)}×{1 + 2 * (3 ** ((i-1) // 2) if i > 0 else 1)}), {hidden_dim}→{hidden_dim} channels'
            } for i in range(n_layers)
        ] + [
            {
                'layer': n_layers + 3,
                'type': 'Output Projection',
                'operation': f'Conv3×3, {hidden_dim}→{img_size[2]} channels'
            }
        ]
    }
}

import json
config_path = os.path.join(OUTPUT_DIR, 'training_config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
logger.info(f"Training configuration saved to: {config_path}")


# ============================================================================
# Visualizing forward process
# ============================================================================

logger.info("Visualizing forward diffusion process...")
model.eval()
for batch_idx, (x, _) in enumerate(test_loader):
    x = x.to(DEVICE)
    perturbed_images, epsilon, pred_epsilon = diffusion(x)
    perturbed_images = diffusion.reverse_scale_to_zero_to_one(perturbed_images)
    break

# Log forward process images to TensorBoard
# Check config integrity before accessing
check_config_integrity()
if writer is not None and get_config_value('log_images', True):
    writer.add_images("Forward_Process/Perturbed", perturbed_images, 0)
    writer.add_image("Forward_Process/Sample_0", perturbed_images[0], 0)
    writer.add_image("Forward_Process/Sample_1", perturbed_images[1], 0)
    writer.add_image("Forward_Process/Sample_2", perturbed_images[2], 0)

    # Log original images for comparison
    writer.add_images("Forward_Process/Original", x[:inference_batch_size], 0)
    writer.add_image("Forward_Process/Original_Sample_0", x[0], 0)

    # Log noise that was added
    noise_visualization = epsilon[:4]  # Take first 4 noise samples
    writer.add_images("Forward_Process/Added_Noise", noise_visualization, 0)
    writer.add_image("Forward_Process/Noise_Sample_0", noise_visualization[0], 0)

# Save forward process images
save_sample_grid(perturbed_images, "Forward Process - Perturbed Images", 
                "forward_process_perturbed.png", FORWARD_PROCESS_DIR)
save_sample_grid(x[:inference_batch_size], "Forward Process - Original Images", 
                "forward_process_original.png", FORWARD_PROCESS_DIR)
save_sample_grid(noise_visualization, "Forward Process - Added Noise", 
                "forward_process_noise.png", FORWARD_PROCESS_DIR)

# Save individual samples
for i in range(min(4, len(perturbed_images))):
    save_single_image(perturbed_images, idx=i, 
                     filename=f"perturbed_sample_{i:02d}.png", 
                     output_dir=FORWARD_PROCESS_DIR)
    save_single_image(x, idx=i, 
                     filename=f"original_sample_{i:02d}.png", 
                     output_dir=FORWARD_PROCESS_DIR)
    save_single_image(noise_visualization, idx=i, 
                     filename=f"noise_sample_{i:02d}.png", 
                     output_dir=FORWARD_PROCESS_DIR)

# Save individual perturbed images
save_single_image(perturbed_images, idx=0, filename="perturbed_image_0.png")
save_single_image(perturbed_images, idx=1, filename="perturbed_image_1.png")
save_single_image(perturbed_images, idx=2, filename="perturbed_image_2.png")


# ============================================================================
# Step 3. Train Denoising Diffusion Probabilistic Models(DDPMs)
# ============================================================================

logger.info("Start training DDPMs...")
model.train()

for epoch in range(epochs):
    noise_prediction_loss = 0
    batch_losses = []
    
    for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        x = x.to(DEVICE)

        noisy_input, epsilon, pred_epsilon = diffusion(x)
        loss = denoising_loss(pred_epsilon, epsilon)

        noise_prediction_loss += loss.item()
        batch_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        
        # Log batch-level metrics every 50 batches
        if batch_idx % 50 == 0 and writer is not None:
            writer.add_scalar("Training/Batch_Loss", loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Training/Learning_Rate_Batch", optimizer.param_groups[0]['lr'], 
                            epoch * len(train_loader) + batch_idx)

    avg_loss = noise_prediction_loss / batch_idx
    logger.info(f"Epoch {epoch + 1}/{epochs} complete! Denoising Loss: {avg_loss:.6f}")
    
    # Calculate additional loss statistics
    batch_losses_tensor = torch.tensor(batch_losses)
    loss_std = batch_losses_tensor.std().item()
    loss_min = batch_losses_tensor.min().item()
    loss_max = batch_losses_tensor.max().item()
    
    # Log training metrics to TensorBoard
    if writer is not None:
        writer.add_scalar("Training/Loss", avg_loss, epoch + 1)
        writer.add_scalar("Training/Loss_Std", loss_std, epoch + 1)
        writer.add_scalar("Training/Loss_Min", loss_min, epoch + 1)
        writer.add_scalar("Training/Loss_Max", loss_max, epoch + 1)
        writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)
        
        # Log loss distribution
        writer.add_histogram("Training/Loss_Distribution", batch_losses_tensor, epoch + 1)
    
    # Save loss curve plot every N epochs
    if (epoch + 1) % get_config_value('loss_analysis_every', 10) == 0:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Loss over batches in this epoch
        plt.subplot(2, 2, 1)
        plt.plot(batch_losses)
        plt.title(f"Loss per Batch - Epoch {epoch + 1}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Loss distribution histogram
        plt.subplot(2, 2, 2)
        plt.hist(batch_losses, bins=20, alpha=0.7, edgecolor='black')
        plt.title(f"Loss Distribution - Epoch {epoch + 1}")
        plt.xlabel("Loss")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Loss statistics
        plt.subplot(2, 2, 3)
        stats = [avg_loss, loss_std, loss_min, loss_max]
        labels = ['Mean', 'Std', 'Min', 'Max']
        plt.bar(labels, stats, alpha=0.7)
        plt.title(f"Loss Statistics - Epoch {epoch + 1}")
        plt.ylabel("Loss Value")
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Learning rate
        plt.subplot(2, 2, 4)
        plt.bar(['Current'], [optimizer.param_groups[0]['lr']], alpha=0.7)
        plt.title(f"Learning Rate - Epoch {epoch + 1}")
        plt.ylabel("Learning Rate")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"loss_analysis_epoch_{epoch+1:03d}.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"Saved loss analysis: loss_analysis_epoch_{epoch+1:03d}.png")
    
    # Log model parameters (gradients and weights) every N epochs
    if (epoch + 1) % get_config_value('param_hist_every', 10) == 0 and writer is not None and get_config_value('log_parameters', True):
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch + 1)
            writer.add_histogram(f"Weights/{name}", param, epoch + 1)
    
    # Log denoising process visualization every N epochs
    if (epoch + 1) % get_config_value('denoising_viz_every', 5) == 0:
        log_denoising_process(diffusion, writer, epoch + 1)
    
    # Save checkpoint every N epochs or at the end
    if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == epochs:
        save_checkpoint(diffusion, optimizer, epoch + 1, avg_loss, CHECKPOINT_DIR)
        
        # Generate samples at checkpoint
        checkpoint_samples = generate_checkpoint_samples(diffusion, epoch + 1, 
                                                       num_samples=get_config_value('checkpoint_num_samples', 16))
        
        # Log checkpoint samples to TensorBoard
        if writer is not None and get_config_value('log_images', True):
            writer.add_images(f"Checkpoint_Samples/Epoch_{epoch+1:03d}", checkpoint_samples, epoch + 1)
            writer.add_image(f"Checkpoint_Samples/Epoch_{epoch+1:03d}_Sample_0", checkpoint_samples[0], epoch + 1)
            writer.add_image(f"Checkpoint_Samples/Epoch_{epoch+1:03d}_Sample_1", checkpoint_samples[1], epoch + 1)
            writer.add_image(f"Checkpoint_Samples/Epoch_{epoch+1:03d}_Sample_2", checkpoint_samples[2], epoch + 1)

logger.info("Training finished!")


# ============================================================================
# Step 4. Sample images from noise.
# ============================================================================

logger.info("Generating images from noise...")
model.eval()

with torch.no_grad():
    generated_images = diffusion.sample(N=get_config_value('final_generation_count', 64))

# Log generated images to TensorBoard
if writer is not None and get_config_value('log_images', True):
    writer.add_images("Generated/Images", generated_images, 0)
    writer.add_image("Generated/Sample_0", generated_images[0], 0)
    writer.add_image("Generated/Sample_1", generated_images[1], 0)
    writer.add_image("Generated/Sample_2", generated_images[2], 0)

# Save individual generated images (if enabled)
if get_config_value('save_generated_individuals', True):
    for i in range(min(6, len(generated_images))):
        save_single_image(generated_images, idx=i, filename=f"generated_image_{i}.png")


# ============================================================================
# Comparison with ground-truth samples
# ============================================================================

logger.info("Saving comparison grids...")
if get_config_value('save_grid_images', True):
    save_sample_grid(perturbed_images, "Perturbed Images", "comparison_perturbed.png")
    save_sample_grid(generated_images, "Generated Images", "comparison_generated.png")
    save_sample_grid(x[:get_config_value('final_generation_count', 64)], "Ground-truth Images", "comparison_groundtruth.png")

# Log comparison images to TensorBoard
if writer is not None and get_config_value('log_images', True):
    writer.add_images("Comparison/Perturbed", perturbed_images, 0)
    writer.add_images("Comparison/Generated", generated_images, 0)
    writer.add_images("Comparison/Ground_Truth", x[:get_config_value('final_generation_count', 64)], 0)

    # Log hyperparameters and metrics to TensorBoard
    writer.add_hparams(hparams, {"Final/Loss": avg_loss})

# Close TensorBoard writer
if writer is not None:
    writer.close()
    logger.info(f"TensorBoard logs saved to: {TENSORBOARD_DIR}")
    logger.info(f"To view TensorBoard: tensorboard --logdir={TENSORBOARD_DIR}")

# Save configuration used for this training run
config_save_path = os.path.join(OUTPUT_DIR, 'training_config_used.json')
with open(config_save_path, 'w') as f:
    json.dump(config, f, indent=2)
logger.info(f"Training configuration saved to: {config_save_path}")

# Copy the original config.ini file to output directory for complete reproducibility
import shutil
config_original_path = os.path.join(OUTPUT_DIR, 'config_original.ini')
try:
    shutil.copy2('config.ini', config_original_path)
    logger.info(f"Original config.ini copied to: {config_original_path}")
except FileNotFoundError:
    logger.warning("Original config.ini file not found - skipping copy")

# Also save a human-readable config summary
config_summary_path = os.path.join(OUTPUT_DIR, 'config_summary.txt')
with open(config_summary_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("DDPM Training Configuration Summary\n")
    f.write("=" * 60 + "\n")
    f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Output directory: {OUTPUT_DIR}\n\n")
    
    f.write("Dataset Configuration:\n")
    f.write(f"  Dataset: {config['dataset']}\n")
    f.write(f"  Image size: {config['img_size']}\n")
    f.write(f"  Dataset path: {config['dataset_path']}\n\n")
    
    f.write("Model Configuration:\n")
    f.write(f"  Timestep embedding dim: {config['timestep_embedding_dim']}\n")
    f.write(f"  Number of layers: {config['n_layers']}\n")
    f.write(f"  Hidden dimension: {config['hidden_dim']}\n")
    f.write(f"  Number of timesteps: {config['n_timesteps']}\n")
    f.write(f"  Beta schedule: {config['beta_minmax']}\n\n")
    
    f.write("Training Configuration:\n")
    f.write(f"  Training batch size: {config['train_batch_size']}\n")
    f.write(f"  Inference batch size: {config['inference_batch_size']}\n")
    f.write(f"  Learning rate: {config['lr']}\n")
    f.write(f"  Number of epochs: {config['epochs']}\n")
    f.write(f"  Save every N epochs: {config['save_every_n_epochs']}\n")
    f.write(f"  Random seed: {config['seed']}\n\n")
    
    f.write("Visualization Configuration:\n")
    f.write(f"  Denoising viz every: {config['denoising_viz_every']} epochs\n")
    f.write(f"  Checkpoint samples every: {config['checkpoint_samples_every']} epochs\n")
    f.write(f"  Checkpoint num samples: {config['checkpoint_num_samples']}\n")
    f.write(f"  Loss analysis every: {config['loss_analysis_every']} epochs\n")
    f.write(f"  Parameter hist every: {config['param_hist_every']} epochs\n\n")
    
    f.write("Output Configuration:\n")
    f.write(f"  Save individual images: {config['save_individual_images']}\n")
    f.write(f"  Save grid images: {config['save_grid_images']}\n")
    f.write(f"  Image DPI: {config['image_dpi']}\n\n")
    
    f.write("Device Configuration:\n")
    f.write(f"  Device: {config['device']}\n")
    f.write(f"  GPU ID: {config['gpu_id']}\n")
    f.write(f"  Actual device used: {DEVICE}\n\n")
    
    f.write("Logging Configuration:\n")
    f.write(f"  Log level: {config['log_level']}\n")
    f.write(f"  Save logs: {config['save_logs']}\n")
    f.write(f"  Save architecture: {config['save_architecture']}\n\n")
    
    f.write("TensorBoard Configuration:\n")
    f.write(f"  Enable TensorBoard: {config['enable_tensorboard']}\n")
    f.write(f"  Log images: {config['log_images']}\n")
    f.write(f"  Log parameters: {config['log_parameters']}\n\n")
    
    f.write("Generation Configuration:\n")
    f.write(f"  Final generation count: {config['final_generation_count']}\n")
    f.write(f"  Checkpoint generation count: {config['checkpoint_generation_count']}\n")
    f.write(f"  Save generated individuals: {config['save_generated_individuals']}\n")
    f.write(f"  Save generated grid: {config['save_generated_grid']}\n\n")
    
    f.write("=" * 60 + "\n")

logger.info(f"Configuration summary saved to: {config_summary_path}")

# Create a config comparison report if original config exists
config_comparison_path = os.path.join(OUTPUT_DIR, 'config_comparison.txt')
try:
    with open('config.ini', 'r') as f:
        original_config_content = f.read()
    
    with open(config_comparison_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Configuration Comparison Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training output: {OUTPUT_DIR}\n\n")
        
        f.write("This report shows the configuration used for this training run.\n")
        f.write("Files saved:\n")
        f.write(f"  - config_original.ini: Original config file used\n")
        f.write(f"  - training_config_used.json: Parsed config as JSON\n")
        f.write(f"  - config_summary.txt: Human-readable summary\n")
        f.write(f"  - config_comparison.txt: This comparison report\n\n")
        
        f.write("To reproduce this training run:\n")
        f.write("1. Copy config_original.ini to your working directory\n")
        f.write("2. Run: python 01_denoising_diffusion_probabilistic_model.py\n\n")
        
        f.write("To generate images from checkpoints:\n")
        f.write("1. Use the saved config: python generate_from_checkpoint.py <checkpoint_path> --config config_original.ini\n")
        f.write("2. Or let it auto-detect: python generate_from_checkpoint.py <checkpoint_path>\n\n")
        
        f.write("=" * 80 + "\n")
    
    logger.info(f"Configuration comparison report saved to: {config_comparison_path}")
except FileNotFoundError:
    logger.warning("Original config.ini not found - skipping comparison report")

logger.info(f"All outputs saved to '{OUTPUT_DIR}/' directory")
logger.info(f"Job completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# Function to generate images from saved checkpoint
# ============================================================================

def generate_from_checkpoint(checkpoint_path, num_images=64, output_dir=None):
    """
    Generate images from a saved checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        num_images: Number of images to generate
        output_dir: Directory to save generated images (default: checkpoint directory)
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Initialize model with same architecture
    model = Denoiser(image_resolution=img_size,
                     hidden_dims=hidden_dims,
                     diffusion_time_embedding_dim=timestep_embedding_dim,
                     n_times=n_timesteps).to(DEVICE)
    
    diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps,
                          beta_minmax=beta_minmax, device=DEVICE).to(DEVICE)
    
    # Load checkpoint
    epoch, loss = load_checkpoint(checkpoint_path, diffusion, device=DEVICE)
    
    # Set output directory
    if output_dir is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        output_dir = os.path.join(checkpoint_dir, f'generated_from_epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating {num_images} images...")
    model.eval()
    
    with torch.no_grad():
        generated_images = diffusion.sample(N=num_images)
    
    # Save individual generated images
    for i in range(min(num_images, 10)):  # Save first 10 as individual images
        save_single_image(generated_images, idx=i, 
                         filename=f"generated_from_epoch_{epoch}_image_{i:02d}.png")
    
    # Save grid
    save_sample_grid(generated_images, f"Generated from Epoch {epoch}", 
                    f"generated_grid_epoch_{epoch}.png")
    
    logger.info(f"Generated images saved to: {output_dir}")
    return generated_images


# Example usage (uncomment to use):
# if __name__ == "__main__":
#     # Generate from latest checkpoint
#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
#     if os.path.exists(latest_checkpoint):
#         generate_from_checkpoint(latest_checkpoint, num_images=32)

