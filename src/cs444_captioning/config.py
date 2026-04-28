from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    project_root: Path = Path.cwd()
    data_root: Path = Path("./data/flickr8k")
    output_root: Path = Path("./outputs")
    encoder_name: str = "google/vit-base-patch16-224-in21k"
    decoder_name: str = "gpt2"
    max_length: int = 40
    batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    generation_num_beams: int = 4
    debug_train_samples: int = 128
    debug_val_samples: int = 64
    debug_test_images: int = 32
    seed: int = 444

    def to_display_dict(self):
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
        return out


def ensure_output_root(cfg: ProjectConfig) -> Path:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    return cfg.output_root

