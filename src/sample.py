import os
import torch
from torchvision.utils import save_image
from model import create_model
from config import TrainConfig

def sample():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(cfg.image_size, cfg.num_timesteps).to(device)

    checkpoint_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    else:
        raise FileNotFoundError("No checkpoint found to load model.")

    model.eval()
    with torch.no_grad():
        sampled_images = model.sample(batch_size=16)
        save_image(sampled_images, os.path.join(cfg.results_dir, "sample.png"), nrow=4, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    sample()
