# DDPM Model Architecture Analysis

## 1. Dimensionality Flow Through the Model

### Input Configuration (MNIST Example)
- **Image Size**: (28, 28, 1) - Height, Width, Channels
- **Batch Size**: 128 (training), 64 (inference)
- **Input Shape**: `[B, C, H, W]` = `[128, 1, 28, 28]`

---

## 2. Conditioning Approach: TIME EMBEDDING ONLY

**Important**: This model uses **ONLY timestep conditioning**. There is:
- ❌ No class conditioning
- ❌ No text conditioning  
- ❌ No other conditional inputs

**Conditioning Method**: Timestep embedding is added to residual blocks using **element-wise addition**

---

## 3. Detailed Tensor Dimensions Throughout the Pipeline

### A. TIMESTEP EMBEDDING PATH

#### Step 1: Sinusoidal Position Embedding
```python
# Input: timestep t
t.shape = [B]  # e.g., [128]

# SinusoidalPosEmb output
embedding_dim = 256
half_dim = 128

# Process:
emb = torch.arange(half_dim, device=device)  # [128]
emb = torch.exp(emb * -log(10000)/(half_dim-1))  # [128]
emb = t[:, None] * emb[None, :]  # [B, 128]
emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, 256]

# Output shape: [B, 256]
```

#### Step 2: Time Projection Network
```python
# Input: [B, 256]
time_embedding = SinusoidalPosEmb(t)  # [B, 256]

# Unsqueeze to create spatial dimensions
time_embedding = time_embedding.unsqueeze(-1).unsqueeze(-2)  # [B, 256, 1, 1]

# Conv1x1 with activation
time_embedding = ConvBlock(256, hidden_dims[0], k=1)(time_embedding)  # [B, 256, 1, 1]
# Conv1x1 without activation  
time_embedding = ConvBlock(256, 256, k=1)(time_embedding)  # [B, 256, 1, 1]

# Final shape: [B, 256, 1, 1] - ready for broadcasting
```

---

### B. IMAGE PATH (Main Forward Process)

#### Input Image
```python
x.shape = [B, C, H, W]  # e.g., [128, 1, 28, 28]
```

#### Step 1: Normalization
```python
x = scale_to_minus_one_to_one(x)  # [128, 1, 28, 28]
# Maps [0, 1] → [-1, 1]
```

#### Step 2: Forward Diffusion (Add Noise)
```python
# Sample random timesteps
t = torch.randint(0, 1000, size=(B,))  # [128]

# Add noise according to schedule
perturbed_x = sqrt_alpha_bar[t] * x + sqrt_one_minus_alpha_bar[t] * epsilon
# Shape: [128, 1, 28, 28]
```

#### Step 3: Input Projection (7x7 Conv)
```python
y = in_project(perturbed_x)  # [128, 1, 28, 28] → [128, 256, 28, 28]
# Expands channels from C=1 to hidden_dim=256
```

#### Step 4: Stacked Convolutional Blocks (n_layers=8)
```python
# Each ConvBlock with residual connection:
for i in range(8):
    # CONDITIONING HAPPENS HERE via element-wise addition:
    x_conditioned = y + time_embedding  # [B, 256, 28, 28] + [B, 256, 1, 1]
                                        # Broadcasting: [B, 256, 28, 28]
    
    y_residual = x_conditioned
    y_conv = Conv3x3(x_conditioned)  # [B, 256, 28, 28]
    y = y_residual + y_conv          # Residual connection
    
    # GroupNorm + SiLU activation
    y = GroupNorm(y)   # [B, 256, 28, 28]
    y = SiLU(y)        # [B, 256, 28, 28]

# Layer-specific dilations:
# Layer 0: dilation=1 (3x3 receptive field)
# Layer 1: dilation=3 (7x7 receptive field)  
# Layer 2: dilation=3 (7x7 receptive field)
# Layer 3: dilation=9 (19x19 receptive field)
# Layer 4: dilation=9 (19x19 receptive field)
# ... exponentially growing receptive fields
```

#### Step 5: Output Projection
```python
y = out_project(y)  # [128, 256, 28, 28] → [128, 1, 28, 28]
# Projects back to original channel dimension C=1
```

#### Output: Predicted Noise
```python
pred_epsilon.shape = [B, C, H, W]  # [128, 1, 28, 28]
```

---

### C. TRAINING LOSS COMPUTATION

```python
# Shapes:
perturbed_images.shape = [128, 1, 28, 28]  # Noisy input
epsilon.shape = [128, 1, 28, 28]            # Ground truth noise
pred_epsilon.shape = [128, 1, 28, 28]       # Predicted noise

# Loss
loss = MSE(pred_epsilon, epsilon)  # Scalar
```

---

### D. SAMPLING/INFERENCE (Reverse Diffusion)

```python
# Start from pure noise
x_t = torch.randn(N, C, H, W)  # [64, 1, 28, 28]

# Iteratively denoise from t=999 → t=0
for t in [999, 998, ..., 1, 0]:
    timestep = torch.tensor([t]).repeat(N)  # [64]
    
    # Predict noise
    epsilon_pred = model(x_t, timestep)  # [64, 1, 28, 28]
    
    # Denoise one step
    mean = (x_t - (1-alpha[t])/sqrt(1-alpha_bar[t]) * epsilon_pred) / sqrt(alpha[t])
    
    if t > 1:
        z = torch.randn_like(x_t)  # [64, 1, 28, 28]
    else:
        z = torch.zeros_like(x_t)  # [64, 1, 28, 28]
    
    x_t = mean + sqrt(beta[t]) * z  # [64, 1, 28, 28]

# Final output
x_0 = reverse_scale_to_zero_to_one(x_t)  # [64, 1, 28, 28]
# Maps [-1, 1] → [0, 1]
```

---

## 4. Key Architectural Choices

### Conditioning Strategy: **Adaptive Group Normalization (AdaGN) - Simplified**
This model uses a **simplified version** of conditional normalization:

```python
# Standard approach (like in Stable Diffusion):
# y = GroupNorm((x - mean) / std) * gamma(t) + beta(t)

# This model's approach:
# x_conditioned = x + time_embedding  # Direct addition
# y = GroupNorm(x_conditioned)
```

**Comparison**:
- ✅ Simpler implementation
- ✅ Fewer parameters
- ❌ Less flexible than AdaGN/AdaLN
- ❌ Time embedding only affects residual path, not normalization parameters

### Receptive Field Growth
```python
# Dilation pattern: 3^(layer//2)
Layer 0-1: dilation=1   → receptive field ≈ 3-5 pixels
Layer 2-3: dilation=3   → receptive field ≈ 7-13 pixels  
Layer 4-5: dilation=9   → receptive field ≈ 19-37 pixels
Layer 6-7: dilation=27  → receptive field ≈ 55+ pixels (covers full image)
```

### Channel Consistency
- All hidden layers maintain **256 channels**
- No U-Net style encoder-decoder with channel changes
- Simpler than typical DDPM implementations

---

## 5. Comparison with Standard DDPM (Ho et al. 2020)

| Aspect | This Implementation | Original DDPM Paper |
|--------|---------------------|---------------------|
| **Architecture** | Stacked Conv + Dilations | U-Net with Attention |
| **Channels** | Constant (256) | Varies (128→512→128) |
| **Conditioning** | Direct Addition | AdaGN (Adaptive GroupNorm) |
| **Spatial Resolution** | Constant (28×28) | Multi-scale (28→14→7→14→28) |
| **Attention** | None | Multi-head attention at 16×16 |
| **Parameters** | ~4.9M | ~35M (for 32×32 images) |

**Trade-offs**:
- ✅ Much simpler and faster to train
- ✅ Good for educational purposes
- ❌ Lower capacity for complex images
- ❌ No multi-scale feature processing

---

## 6. Variance Schedule (Beta Schedule)

```python
# Linear schedule from beta_min to beta_max
beta_1 = 1e-4 = 0.0001
beta_T = 2e-2 = 0.02
betas = linspace(0.0001, 0.02, steps=1000)

# Alpha values
alphas = 1 - betas  # [0.9999, 0.9998, ..., 0.98]

# Cumulative product (alpha_bar)
alpha_bars = cumprod(alphas)  # [0.9999, 0.9997, ..., 0.000X]
```

**Noise Schedule Interpretation**:
- t=0: Almost no noise (alpha_bar ≈ 1.0)
- t=500: Moderate noise (alpha_bar ≈ 0.1-0.2)
- t=999: Pure noise (alpha_bar ≈ 0.0001)


