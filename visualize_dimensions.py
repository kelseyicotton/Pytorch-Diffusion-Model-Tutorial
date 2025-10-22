"""
Visualize tensor dimensions flowing through the DDPM model
Run this to see actual shapes at each layer
"""

import torch
import torch.nn as nn
import math
import logging
from datetime import datetime

# Setup logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Simplified version of the model components for dimension tracking
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


def visualize_dimensions():
    """
    Trace tensor dimensions through the DDPM pipeline
    """
    logger.info("="*80)
    logger.info("DDPM TENSOR DIMENSION FLOW (MNIST Example)")
    logger.info("="*80)
    
    # Configuration
    B = 8  # Batch size (smaller for visualization)
    C = 1  # Channels (MNIST is grayscale)
    H, W = 28, 28  # Image dimensions
    hidden_dim = 256
    time_emb_dim = 256
    n_timesteps = 1000
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Batch size (B): {B}")
    logger.info(f"   Channels (C): {C}")
    logger.info(f"   Height (H): {H}")
    logger.info(f"   Width (W): {W}")
    logger.info(f"   Hidden dim: {hidden_dim}")
    logger.info(f"   Time embedding dim: {time_emb_dim}")
    logger.info(f"   Timesteps: {n_timesteps}")
    
    # ========================================================================
    # 1. TIME EMBEDDING PATH
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("1️⃣  TIME EMBEDDING PATH")
    logger.info("="*80)
    
    # Random timestep
    t = torch.randint(0, n_timesteps, (B,))
    logger.info(f"\n   Input timestep t: {t.shape} = {list(t.shape)}")
    
    # Sinusoidal embedding
    time_emb_model = SinusoidalPosEmb(time_emb_dim)
    time_emb = time_emb_model(t)
    logger.info(f"   ↓ SinusoidalPosEmb")
    logger.info(f"   Time embedding: {time_emb.shape} = {list(time_emb.shape)}")
    
    # Unsqueeze for spatial broadcast
    time_emb_spatial = time_emb.unsqueeze(-1).unsqueeze(-2)
    logger.info(f"   ↓ Unsqueeze (add spatial dims)")
    logger.info(f"   Time embedding (spatial): {time_emb_spatial.shape} = {list(time_emb_spatial.shape)}")
    
    # After time projection (2 Conv1x1 layers)
    logger.info(f"   ↓ Time Projection Network (Conv1x1 x2)")
    logger.info(f"   Final time embedding: [{B}, {hidden_dim}, 1, 1]")
    
    # ========================================================================
    # 2. IMAGE PATH - FORWARD DIFFUSION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("2️⃣  IMAGE PATH - FORWARD DIFFUSION (Adding Noise)")
    logger.info("="*80)
    
    # Original image
    x = torch.randn(B, C, H, W)
    logger.info(f"\n   Input image x_0: {x.shape} = {list(x.shape)}")
    
    # Normalize to [-1, 1]
    x_normalized = x * 2 - 1
    logger.info(f"   ↓ Normalize to [-1, 1]")
    logger.info(f"   Normalized: {x_normalized.shape} = {list(x_normalized.shape)}")
    
    # Add noise (forward diffusion)
    epsilon = torch.randn_like(x_normalized)
    alpha_bar_t = 0.5  # Example value
    x_t = torch.sqrt(torch.tensor(alpha_bar_t)) * x_normalized + \
          torch.sqrt(torch.tensor(1 - alpha_bar_t)) * epsilon
    logger.info(f"   ↓ Add noise: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε")
    logger.info(f"   Noisy image x_t: {x_t.shape} = {list(x_t.shape)}")
    
    # ========================================================================
    # 3. DENOISER NETWORK
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("3️⃣  DENOISER NETWORK (Predicting Noise)")
    logger.info("="*80)
    
    # Input projection
    logger.info(f"\n   Input: {x_t.shape} = {list(x_t.shape)}")
    y = torch.randn(B, hidden_dim, H, W)  # Simulating Conv7x7
    logger.info(f"   ↓ Input Projection (Conv 7×7, {C}→{hidden_dim})")
    logger.info(f"   Features: {y.shape} = {list(y.shape)}")
    
    # Convolutional blocks with time conditioning
    logger.info(f"\n   ┌─ Residual Blocks with Time Conditioning ─────────────────")
    for i in range(8):
        dilation = 3 ** ((i-1) // 2) if i > 0 else 1
        receptive_field = 1 + 2 * dilation  # Approximate for 3x3 kernel
        
        logger.info(f"   │ Layer {i}:")
        logger.info(f"   │   Features: [{B}, {hidden_dim}, {H}, {W}]")
        logger.info(f"   │   ↓ Add time embedding (broadcast)")
        logger.info(f"   │   Conditioned: [{B}, {hidden_dim}, {H}, {W}] + [{B}, {hidden_dim}, 1, 1]")
        logger.info(f"   │            = [{B}, {hidden_dim}, {H}, {W}]")
        logger.info(f"   │   ↓ Conv 3×3 (dilation={dilation}, receptive_field≈{receptive_field}×{receptive_field})")
        logger.info(f"   │   ↓ Residual connection")
        logger.info(f"   │   ↓ GroupNorm + SiLU")
        logger.info(f"   │   Output: [{B}, {hidden_dim}, {H}, {W}]")
        if i < 7:
            logger.info(f"   │")
    logger.info(f"   └──────────────────────────────────────────────────────────")
    
    # Output projection
    pred_epsilon = torch.randn(B, C, H, W)  # Simulating Conv3x3
    logger.info(f"\n   ↓ Output Projection (Conv 3×3, {hidden_dim}→{C})")
    logger.info(f"   Predicted noise ε̂: {pred_epsilon.shape} = {list(pred_epsilon.shape)}")
    
    # ========================================================================
    # 4. LOSS COMPUTATION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("4️⃣  LOSS COMPUTATION")
    logger.info("="*80)
    
    logger.info(f"\n   Ground truth noise ε: {epsilon.shape} = {list(epsilon.shape)}")
    logger.info(f"   Predicted noise ε̂:    {pred_epsilon.shape} = {list(pred_epsilon.shape)}")
    logger.info(f"   ↓ MSE Loss")
    logger.info(f"   Loss: scalar = ||ε - ε̂||²")
    
    # ========================================================================
    # 5. REVERSE DIFFUSION (SAMPLING)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("5️⃣  REVERSE DIFFUSION (Sampling/Inference)")
    logger.info("="*80)
    
    logger.info(f"\n   Start from pure noise:")
    x_T = torch.randn(B, C, H, W)
    logger.info(f"   x_T ~ N(0, I): {x_T.shape} = {list(x_T.shape)}")
    
    logger.info(f"\n   ┌─ Iterative Denoising Loop ───────────────────────────────")
    logger.info(f"   │ for t in [999, 998, 997, ..., 2, 1, 0]:")
    logger.info(f"   │   ")
    logger.info(f"   │   Current noisy image: [{B}, {C}, {H}, {W}]")
    logger.info(f"   │   ↓ Denoiser Network(x_t, t)")
    logger.info(f"   │   Predicted noise: [{B}, {C}, {H}, {W}]")
    logger.info(f"   │   ↓ Compute mean of x_{{t-1}}")
    logger.info(f"   │   μ_{{t-1}} = (x_t - (1-α_t)/√(1-ᾱ_t) · ε̂) / √α_t")
    logger.info(f"   │   ↓ Add noise (except at t=0)")
    logger.info(f"   │   x_{{t-1}} = μ_{{t-1}} + √β_t · z  (z ~ N(0,I))")
    logger.info(f"   │   Next state: [{B}, {C}, {H}, {W}]")
    logger.info(f"   │")
    logger.info(f"   └──────────────────────────────────────────────────────────")
    
    logger.info(f"\n   Final denoised image x_0: {x_T.shape} = {list(x_T.shape)}")
    logger.info(f"   ↓ Denormalize [-1, 1] → [0, 1]")
    logger.info(f"   Output image: {x_T.shape} = {list(x_T.shape)}")
    
    # ========================================================================
    # 6. BROADCASTING EXPLANATION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("6️⃣  KEY CONCEPT: BROADCASTING (How Time Conditioning Works)")
    logger.info("="*80)
    
    logger.info(f"\n   Image features:     [{B}, {hidden_dim}, {H}, {W}]")
    logger.info(f"   Time embedding:     [{B}, {hidden_dim},  1,  1]")
    logger.info(f"                          ↓  Broadcasting happens here  ↓")
    logger.info(f"   Time (broadcast):   [{B}, {hidden_dim}, {H}, {W}]  (same value for all spatial positions)")
    logger.info(f"   ────────────────────────────────────────────────────")
    logger.info(f"   Sum:                [{B}, {hidden_dim}, {H}, {W}]")
    
    logger.info(f"\n   💡 The time embedding [{B}, {hidden_dim}, 1, 1] is broadcast to")
    logger.info(f"      [{B}, {hidden_dim}, {H}, {W}] by repeating the same values")
    logger.info(f"      across all spatial positions (H×W grid).")
    
    # ========================================================================
    # 7. PARAMETER COUNT
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("7️⃣  PARAMETER COUNT BREAKDOWN")
    logger.info("="*80)
    
    logger.info(f"\n   Time Embedding:")
    logger.info(f"   • SinusoidalPosEmb: 0 (no learnable params)")
    logger.info(f"   • Time Projection: ~{2 * time_emb_dim * hidden_dim:,} params")
    
    logger.info(f"\n   Image Path:")
    logger.info(f"   • Input Projection (7×7 conv): ~{49 * C * hidden_dim:,} params")
    logger.info(f"   • 8 Conv Blocks (3×3 conv): ~{8 * 9 * hidden_dim * hidden_dim:,} params")
    logger.info(f"   • GroupNorm layers: ~{8 * 2 * hidden_dim:,} params")
    logger.info(f"   • Output Projection (3×3 conv): ~{9 * hidden_dim * C:,} params")
    
    logger.info(f"\n   📊 Total: ~4.9M parameters")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Script completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")


if __name__ == "__main__":
    logger.info(f"Starting dimension visualization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    visualize_dimensions()

