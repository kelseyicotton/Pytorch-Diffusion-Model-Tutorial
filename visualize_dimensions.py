"""
Visualize tensor dimensions flowing through the DDPM model
Run this to see actual shapes at each layer
"""

import torch
import torch.nn as nn
import math

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
    print("="*80)
    print("DDPM TENSOR DIMENSION FLOW (MNIST Example)")
    print("="*80)
    
    # Configuration
    B = 8  # Batch size (smaller for visualization)
    C = 1  # Channels (MNIST is grayscale)
    H, W = 28, 28  # Image dimensions
    hidden_dim = 256
    time_emb_dim = 256
    n_timesteps = 1000
    
    print(f"\n📋 Configuration:")
    print(f"   Batch size (B): {B}")
    print(f"   Channels (C): {C}")
    print(f"   Height (H): {H}")
    print(f"   Width (W): {W}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Time embedding dim: {time_emb_dim}")
    print(f"   Timesteps: {n_timesteps}")
    
    # ========================================================================
    # 1. TIME EMBEDDING PATH
    # ========================================================================
    print("\n" + "="*80)
    print("1️⃣  TIME EMBEDDING PATH")
    print("="*80)
    
    # Random timestep
    t = torch.randint(0, n_timesteps, (B,))
    print(f"\n   Input timestep t: {t.shape} = {list(t.shape)}")
    
    # Sinusoidal embedding
    time_emb_model = SinusoidalPosEmb(time_emb_dim)
    time_emb = time_emb_model(t)
    print(f"   ↓ SinusoidalPosEmb")
    print(f"   Time embedding: {time_emb.shape} = {list(time_emb.shape)}")
    
    # Unsqueeze for spatial broadcast
    time_emb_spatial = time_emb.unsqueeze(-1).unsqueeze(-2)
    print(f"   ↓ Unsqueeze (add spatial dims)")
    print(f"   Time embedding (spatial): {time_emb_spatial.shape} = {list(time_emb_spatial.shape)}")
    
    # After time projection (2 Conv1x1 layers)
    print(f"   ↓ Time Projection Network (Conv1x1 x2)")
    print(f"   Final time embedding: [{B}, {hidden_dim}, 1, 1]")
    
    # ========================================================================
    # 2. IMAGE PATH - FORWARD DIFFUSION
    # ========================================================================
    print("\n" + "="*80)
    print("2️⃣  IMAGE PATH - FORWARD DIFFUSION (Adding Noise)")
    print("="*80)
    
    # Original image
    x = torch.randn(B, C, H, W)
    print(f"\n   Input image x_0: {x.shape} = {list(x.shape)}")
    
    # Normalize to [-1, 1]
    x_normalized = x * 2 - 1
    print(f"   ↓ Normalize to [-1, 1]")
    print(f"   Normalized: {x_normalized.shape} = {list(x_normalized.shape)}")
    
    # Add noise (forward diffusion)
    epsilon = torch.randn_like(x_normalized)
    alpha_bar_t = 0.5  # Example value
    x_t = torch.sqrt(torch.tensor(alpha_bar_t)) * x_normalized + \
          torch.sqrt(torch.tensor(1 - alpha_bar_t)) * epsilon
    print(f"   ↓ Add noise: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε")
    print(f"   Noisy image x_t: {x_t.shape} = {list(x_t.shape)}")
    
    # ========================================================================
    # 3. DENOISER NETWORK
    # ========================================================================
    print("\n" + "="*80)
    print("3️⃣  DENOISER NETWORK (Predicting Noise)")
    print("="*80)
    
    # Input projection
    print(f"\n   Input: {x_t.shape} = {list(x_t.shape)}")
    y = torch.randn(B, hidden_dim, H, W)  # Simulating Conv7x7
    print(f"   ↓ Input Projection (Conv 7×7, {C}→{hidden_dim})")
    print(f"   Features: {y.shape} = {list(y.shape)}")
    
    # Convolutional blocks with time conditioning
    print(f"\n   ┌─ Residual Blocks with Time Conditioning ─────────────────")
    for i in range(8):
        dilation = 3 ** ((i-1) // 2) if i > 0 else 1
        receptive_field = 1 + 2 * dilation  # Approximate for 3x3 kernel
        
        print(f"   │ Layer {i}:")
        print(f"   │   Features: [{B}, {hidden_dim}, {H}, {W}]")
        print(f"   │   ↓ Add time embedding (broadcast)")
        print(f"   │   Conditioned: [{B}, {hidden_dim}, {H}, {W}] + [{B}, {hidden_dim}, 1, 1]")
        print(f"   │            = [{B}, {hidden_dim}, {H}, {W}]")
        print(f"   │   ↓ Conv 3×3 (dilation={dilation}, receptive_field≈{receptive_field}×{receptive_field})")
        print(f"   │   ↓ Residual connection")
        print(f"   │   ↓ GroupNorm + SiLU")
        print(f"   │   Output: [{B}, {hidden_dim}, {H}, {W}]")
        if i < 7:
            print(f"   │")
    print(f"   └──────────────────────────────────────────────────────────")
    
    # Output projection
    pred_epsilon = torch.randn(B, C, H, W)  # Simulating Conv3x3
    print(f"\n   ↓ Output Projection (Conv 3×3, {hidden_dim}→{C})")
    print(f"   Predicted noise ε̂: {pred_epsilon.shape} = {list(pred_epsilon.shape)}")
    
    # ========================================================================
    # 4. LOSS COMPUTATION
    # ========================================================================
    print("\n" + "="*80)
    print("4️⃣  LOSS COMPUTATION")
    print("="*80)
    
    print(f"\n   Ground truth noise ε: {epsilon.shape} = {list(epsilon.shape)}")
    print(f"   Predicted noise ε̂:    {pred_epsilon.shape} = {list(pred_epsilon.shape)}")
    print(f"   ↓ MSE Loss")
    print(f"   Loss: scalar = ||ε - ε̂||²")
    
    # ========================================================================
    # 5. REVERSE DIFFUSION (SAMPLING)
    # ========================================================================
    print("\n" + "="*80)
    print("5️⃣  REVERSE DIFFUSION (Sampling/Inference)")
    print("="*80)
    
    print(f"\n   Start from pure noise:")
    x_T = torch.randn(B, C, H, W)
    print(f"   x_T ~ N(0, I): {x_T.shape} = {list(x_T.shape)}")
    
    print(f"\n   ┌─ Iterative Denoising Loop ───────────────────────────────")
    print(f"   │ for t in [999, 998, 997, ..., 2, 1, 0]:")
    print(f"   │   ")
    print(f"   │   Current noisy image: [{B}, {C}, {H}, {W}]")
    print(f"   │   ↓ Denoiser Network(x_t, t)")
    print(f"   │   Predicted noise: [{B}, {C}, {H}, {W}]")
    print(f"   │   ↓ Compute mean of x_{{t-1}}")
    print(f"   │   μ_{{t-1}} = (x_t - (1-α_t)/√(1-ᾱ_t) · ε̂) / √α_t")
    print(f"   │   ↓ Add noise (except at t=0)")
    print(f"   │   x_{{t-1}} = μ_{{t-1}} + √β_t · z  (z ~ N(0,I))")
    print(f"   │   Next state: [{B}, {C}, {H}, {W}]")
    print(f"   │")
    print(f"   └──────────────────────────────────────────────────────────")
    
    print(f"\n   Final denoised image x_0: {x_T.shape} = {list(x_T.shape)}")
    print(f"   ↓ Denormalize [-1, 1] → [0, 1]")
    print(f"   Output image: {x_T.shape} = {list(x_T.shape)}")
    
    # ========================================================================
    # 6. BROADCASTING EXPLANATION
    # ========================================================================
    print("\n" + "="*80)
    print("6️⃣  KEY CONCEPT: BROADCASTING (How Time Conditioning Works)")
    print("="*80)
    
    print(f"\n   Image features:     [{B}, {hidden_dim}, {H}, {W}]")
    print(f"   Time embedding:     [{B}, {hidden_dim},  1,  1]")
    print(f"                          ↓  Broadcasting happens here  ↓")
    print(f"   Time (broadcast):   [{B}, {hidden_dim}, {H}, {W}]  (same value for all spatial positions)")
    print(f"   ────────────────────────────────────────────────────")
    print(f"   Sum:                [{B}, {hidden_dim}, {H}, {W}]")
    
    print(f"\n   💡 The time embedding [{B}, {hidden_dim}, 1, 1] is broadcast to")
    print(f"      [{B}, {hidden_dim}, {H}, {W}] by repeating the same values")
    print(f"      across all spatial positions (H×W grid).")
    
    # ========================================================================
    # 7. PARAMETER COUNT
    # ========================================================================
    print("\n" + "="*80)
    print("7️⃣  PARAMETER COUNT BREAKDOWN")
    print("="*80)
    
    print(f"\n   Time Embedding:")
    print(f"   • SinusoidalPosEmb: 0 (no learnable params)")
    print(f"   • Time Projection: ~{2 * time_emb_dim * hidden_dim:,} params")
    
    print(f"\n   Image Path:")
    print(f"   • Input Projection (7×7 conv): ~{49 * C * hidden_dim:,} params")
    print(f"   • 8 Conv Blocks (3×3 conv): ~{8 * 9 * hidden_dim * hidden_dim:,} params")
    print(f"   • GroupNorm layers: ~{8 * 2 * hidden_dim:,} params")
    print(f"   • Output Projection (3×3 conv): ~{9 * hidden_dim * C:,} params")
    
    print(f"\n   📊 Total: ~4.9M parameters")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print()


if __name__ == "__main__":
    visualize_dimensions()

