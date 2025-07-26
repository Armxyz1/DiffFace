from dataclasses import dataclass

@dataclass
class TrainConfig:
    image_size: int = 128
    channels: int = 3
    batch_size: int = 16
    num_epochs: int = 500
    lr: float = 1e-4
    save_every: int = 1
    ema_decay: float = 0.995
    num_timesteps: int = 1000
    results_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"
    dataset_path: str = "./dataset"
    use_mixed_precision: bool = True
    seed: int = 42
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    min_delta: float = 1e-4