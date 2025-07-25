import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from accelerate import Accelerator
from tqdm import tqdm

from dataset import FaceDataset
from model import create_model
from utils import set_seed, save_checkpoint, load_checkpoint
from config import TrainConfig

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images in val_loader:
            loss = model(images)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    accelerator = Accelerator(mixed_precision="fp16" if cfg.use_mixed_precision else "no")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Load and split dataset
    dataset = FaceDataset(cfg.dataset_path, cfg.image_size)
    val_size = int(cfg.validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # Model and optimizer
    model = create_model(cfg.image_size, cfg.num_timesteps)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # Resume from checkpoint if available
    start_epoch = 0
    latest_ckpt = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        print("Resuming from checkpoint...")
        start_epoch = load_checkpoint(model, optimizer, latest_ckpt)

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        for images in pbar:
            loss = model(images)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item())

        # Validation
        val_loss = evaluate(model, val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss + cfg.min_delta < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            if accelerator.is_main_process:
                save_checkpoint(accelerator.unwrap_model(model), optimizer, epoch+1,
                                os.path.join(cfg.checkpoint_dir, "best.pt"))
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{cfg.early_stopping_patience}")
            if patience_counter >= cfg.early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Save latest checkpoint
        if accelerator.is_main_process:
            save_checkpoint(accelerator.unwrap_model(model), optimizer, epoch+1, latest_ckpt)

if __name__ == "__main__":
    train()
