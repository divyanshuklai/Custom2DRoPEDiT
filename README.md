# RoPE-DiT: Rectified Flow Transformers with 2D Rotary Embeddings

This project implements a **Diffusion Transformer (DiT)** architecture trained on **CIFAR-10** using **Rectified Flow** matching objective. It features a custom implementation of **2D Rotary Positional Embeddings (RoPE)** designed specifically for image data.

![Generated Samples](assets/generated_samples.png)
*Generated samples from the model trained on CIFAR-10 (steps=100, CFG=3.0)*

## Key Features

### 1. 2D Rotary Positional Embeddings (RoPE)
Unlike standard absolute positional embeddings or 1D RoPE applied to flattened sequences, this model implements a true **2D RoPE**.
- **Mechanism**: The attention head dimension is split into two halves. One half encodes vertical positions (rows), and the other encodes horizontal positions (columns).
- **Implementation**: See `src/models/rope_dit_modelling.py` (`RoPE2DMHA`).
    - Frequencies are computed for both spatial dimensions.
    - During attention computation, queries and keys are rotated in the complex plane corresponding to their 2D spatial locations.
    - This allows the model to better capture relative 2D distances between patches.

### 2. Rectified Flow Training
The model is trained using the **Rectified Flow** method, which learns a straight-line probability flow between the noise distribution $\mathcal{N}(0, I)$ and the data distribution.

- **Objective**: Minimize the Mean Squared Error (MSE) between the model output and the velocity field $v = X_1 - X_0$.
- **Loss**: $L(\theta) = E_{t, x_0, x_1} [ || v_\theta(x_t, t) - (x_1 - x_0) ||^2 ]$
- **Interpolation**: $x_t = t x_1 + (1-t) x_0$
- This results in straighter trajectories during sampling compared to standard DDPM, allowing for efficient generation with fewer steps (e.g., using Euler solver).

### 3. Architecture Details
- **Backbone**: Vision Transformer with Adaptive Layer Normalization (AdaLN-Zero) for conditioning.
- **Conditioning**:
    - **Timestep**: Sinusoidal embeddings processed via MLP.
    - **Class Labels**: Learned embeddings overlaid with timestep embeddings.
    - **Conditioning Mechanism**: `AdaLN-Zero` modulates the normalization parameters ($\gamma, \beta$) and scales residual connections ($\alpha$) based on class and time info.
- **Patchification**: Images are patchified (e.g., 4x4 patches) and projected to the model dimension.

## Project Structure

```
├── src/
│   ├── models/
│   │   └── rope_dit_modelling.py  # RoPE-DiT Architecture & 2D RoPE implementation
│   ├── engine/
│   │   └── trainer.py             # Rectified Flow Trainer
│   └── utils/
│       └── sampler.py             # Euler Sampler with CFG
├── notebooks/
│   └── test.ipynb                 # Training and Testing notebook
├── trained/                       # Checkpoints
└── assets/                        # Generated images
```

## Model Configuration (Default)

- **Model Dimension**: 256
- **Number of Blocks**: 6
- **Attention Heads**: 8 (32 dim per head)
- **Patch Size**: 4
- **Parameters**: ~4-5M (Small scale for CIFAR-10 demonstration)

## Usage

### Training & Inference
The project is set up to run primarily through the `test.ipynb` notebook.

1.  **Dependencies**: Install PyTorch, torchvision, wandb, matplotlib, tqdm, torchinfo.
2.  **Dataset**: CIFAR-10 (automatically downloaded).
3.  **Training**: The notebook contains the training loop using `RectifiedFlowTrainer`.
4.  **Sampling**: Use `euler_sampler` from `src.utils.sampler` with `forwardCFG` for Classifier-Free Guidance.

```python
# Inference Example
from src.utils.sampler import euler_sampler
x = euler_sampler(model, x0, y, step_size=0.01, num_steps=100, cfg_scale=3.0)
```

## Results

The model successfully generates coherent CIFAR-10 like images. The Rectified Flow objective ensures that the generation process is stable and efficient. The 2D RoPE supposedly helps in maintaining spatial consistency better than learnable absolute embeddings for patch-based models.
