import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def create_model(image_size, num_timesteps):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=num_timesteps,
    )

    return diffusion
