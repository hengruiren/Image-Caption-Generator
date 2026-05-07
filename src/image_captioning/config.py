from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_root: Path = Path("/workspace/data/coco")
    output_root: Path = Path("/workspace/outputs")

    # Data
    train_subset_size: int = 30_000
    val_size: int = 5_000
    seed: int = 42

    # Model
    decoder_name: str = "gpt2"
    max_length: int = 40

    # Training
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    num_workers: int = 4

    # Generation
    num_beams: int = 4

    # Experiments
    experiments: list = field(default_factory=lambda: [
        {"name": "vit_no_mapper",   "encoder": "google/vit-base-patch16-224-in21k", "use_mapper": False},
        {"name": "vit_mlp_mapper",  "encoder": "google/vit-base-patch16-224-in21k", "use_mapper": True},
        {"name": "clip_no_mapper",  "encoder": "openai/clip-vit-base-patch16",       "use_mapper": False},
        {"name": "clip_mlp_mapper", "encoder": "openai/clip-vit-base-patch16",       "use_mapper": True},
    ])

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    @property
    def train_image_dir(self):
        return self.data_root / "train2017"

    @property
    def val_image_dir(self):
        return self.data_root / "val2017"

    @property
    def train_ann_file(self):
        return self.data_root / "annotations" / "captions_train2017.json"

    @property
    def val_ann_file(self):
        return self.data_root / "annotations" / "captions_val2017.json"
