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

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math


# Create timestamped output directory first
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f'outputs_{timestamp}'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Setup logging with timestamp (both console and file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log'))  # File output
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Created output directory: {OUTPUT_DIR}")
logger.info(f"Created checkpoint directory: {CHECKPOINT_DIR}")


# ============================================================================
# Model Hyperparameters
# ============================================================================

dataset_path = '~/datasets'

cuda = torch.cuda.is_available()  # Automatically detect GPU availability
DEVICE = torch.device("cuda:0" if cuda else "cpu")
logger.info(f"Using device: {DEVICE}")

dataset = 'MNIST'
img_size = (32, 32, 3) if dataset == "CIFAR10" else (28, 28, 1)  # (width, height, channels)

timestep_embedding_dim = 256
n_layers = 8
hidden_dim = 256
n_timesteps = 1000
beta_minmax = [1e-4, 2e-2]

train_batch_size = 128
inference_batch_size = 64
lr = 5e-5
epochs = 100
save_every_n_epochs = 10  # Save checkpoint every N epochs

seed = 1234

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

# Training configuration will be saved after model initialization


# ============================================================================
# Step 1. Load (or download) Dataset
# ============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])

kwargs = {'num_workers': 1, 'pin_memory': True}

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

def save_single_image(x, idx, filename):
    """Save a single image from a batch"""
    fig = plt.figure()
    plt.imshow(x[idx].transpose(0, 1).transpose(1, 2).detach().cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {filename}")


def save_sample_grid(x, postfix, filename):
    """Save a grid of sample images"""
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {filename}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        x = x.to(DEVICE)

        noisy_input, epsilon, pred_epsilon = diffusion(x)
        loss = denoising_loss(pred_epsilon, epsilon)

        noise_prediction_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = noise_prediction_loss / batch_idx
    logger.info(f"Epoch {epoch + 1}/{epochs} complete! Denoising Loss: {avg_loss:.6f}")
    
    # Save checkpoint every N epochs or at the end
    if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == epochs:
        save_checkpoint(diffusion, optimizer, epoch + 1, avg_loss, CHECKPOINT_DIR)

logger.info("Training finished!")


# ============================================================================
# Step 4. Sample images from noise.
# ============================================================================

logger.info("Generating images from noise...")
model.eval()

with torch.no_grad():
    generated_images = diffusion.sample(N=inference_batch_size)

# Save individual generated images
save_single_image(generated_images, idx=0, filename="generated_image_0.png")
save_single_image(generated_images, idx=1, filename="generated_image_1.png")
save_single_image(generated_images, idx=2, filename="generated_image_2.png")
save_single_image(generated_images, idx=3, filename="generated_image_3.png")
save_single_image(generated_images, idx=4, filename="generated_image_4.png")
save_single_image(generated_images, idx=5, filename="generated_image_5.png")


# ============================================================================
# Comparison with ground-truth samples
# ============================================================================

logger.info("Saving comparison grids...")
save_sample_grid(perturbed_images, "Perturbed Images", "comparison_perturbed.png")
save_sample_grid(generated_images, "Generated Images", "comparison_generated.png")
save_sample_grid(x[:inference_batch_size], "Ground-truth Images", "comparison_groundtruth.png")

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

