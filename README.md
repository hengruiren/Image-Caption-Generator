# Image Caption Generator

CS444 final project on cross-modal alignment for image captioning.

This project studies how to connect a pretrained vision encoder with a pretrained language decoder for image caption generation. The current baseline uses a ViT image encoder and GPT-2 text decoder through cross-attention. The planned experiments compare whether pretrained image-text alignment from CLIP and an explicit MLP mapping module improve caption quality.

## Project Goal

Image captioning requires a model to understand visual content and generate a fluent natural-language description. A central challenge is that pretrained vision models and pretrained language models usually live in different representation spaces.

The main research question is:

> In encoder-decoder multimodal architectures, how can we most effectively align the representation spaces of pretrained vision encoders and pretrained language models?

## Experiment Plan

The core experiment matrix is:

| Experiment | Encoder | Mapping module | Decoder |
| --- | --- | --- | --- |
| `vit_no_mapper` | `google/vit-base-patch16-224-in21k` | None | `gpt2` |
| `vit_mlp_mapper` | `google/vit-base-patch16-224-in21k` | 2-layer MLP | `gpt2` |
| `clip_no_mapper` | `openai/clip-vit-base-patch16` | None | `gpt2` |
| `clip_mlp_mapper` | `openai/clip-vit-base-patch16` | 2-layer MLP | `gpt2` |

Minimum target:

- Train and evaluate all four core experiments on Flickr8k.
- Compare BLEU-4, METEOR, and CIDEr.
- Analyze the effect of CLIP pre-alignment and explicit MLP mapping.

Stretch goals:

- Add LoRA fine-tuning for GPT-2.
- Run data-size ablations with smaller Flickr8k subsets.
- Generate attention visualizations for qualitative analysis.

## Current Status

Implemented:

- Flickr8k caption loading and image-level train/validation/test split.
- PyTorch dataset and batch collation.
- ViT + GPT-2 baseline using Hugging Face `VisionEncoderDecoderModel`.
- Frozen encoder training loop with gradient accumulation and checkpoint saving.
- Caption generation.
- BLEU-4, METEOR, and CIDEr evaluation helpers.
- Basic visualization utilities for samples, predictions, and metric plots.

In progress / planned:

- CLIP ViT encoder condition.
- MLP mapping module between encoder and decoder.
- LoRA fine-tuning condition.
- Full experiment comparison table.

## Repository Structure

```text
.
├── CS444_Final_Project_Image_Captioning_Alignment.ipynb
├── CS444 Progress Update.pdf
├── README.md
└── src/
    └── cs444_captioning/
        ├── config.py
        ├── data.py
        ├── evaluation.py
        ├── generation.py
        ├── modeling.py
        ├── training.py
        ├── utils.py
        └── visualization.py
```

## Dataset

Primary dataset: Flickr8k.

Expected local layout:

```text
data/flickr8k/
├── captions.txt
└── Images/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

The data loader also tries to auto-detect common Kaggle layouts such as:

- `/kaggle/input/flickr8k`
- `/kaggle/input/flickr8k-dataset`
- `./flickr8k`

Dataset source:

- Kaggle Flickr8k: https://www.kaggle.com/datasets/adityajn105/flickr8k

## Setup

Install the main Python dependencies:

```bash
pip install torch torchvision transformers pandas numpy pillow tqdm matplotlib seaborn evaluate
```

For CIDEr evaluation:

```bash
pip install pycocoevalcap
```

Optional, for LoRA experiments:

```bash
pip install peft
```

## Usage

The main workflow is in:

```text
CS444_Final_Project_Image_Captioning_Alignment.ipynb
```

Typical notebook flow:

1. Import project modules from `src/cs444_captioning`.
2. Create a `ProjectConfig`.
3. Load Flickr8k captions and image files.
4. Build train/validation/test datasets.
5. Build the ViT + GPT-2 baseline model.
6. Train the model.
7. Generate predictions on the test split.
8. Evaluate predictions with BLEU-4, METEOR, and CIDEr.

Core baseline construction:

```python
from cs444_captioning.config import ProjectConfig
from cs444_captioning.modeling import load_processors, build_vit_gpt2_baseline
from cs444_captioning.utils import describe_device

cfg = ProjectConfig()
device = describe_device()
image_processor, tokenizer = load_processors(cfg)
model = build_vit_gpt2_baseline(cfg, tokenizer, device, freeze_encoder=True)
```

## Outputs

Training and evaluation artifacts are written under:

```text
outputs/<experiment_name>/
```

Common output files:

- `training_log.csv`
- `checkpoint_last/`
- `predictions.json`

## Evaluation Metrics

- BLEU-4: measures 4-gram overlap between generated and reference captions.
- METEOR: measures caption similarity with recall and synonym-aware matching.
- CIDEr: measures consensus similarity against multiple human references.

## Implementation Roadmap

Recommended next steps:

1. Keep `vit_no_mapper` as the controlled baseline.
2. Add a config option for encoder selection and run `clip_no_mapper`.
3. Implement `MLPMapper` and wrap the encoder output before GPT-2 cross-attention.
4. Run `vit_mlp_mapper` and `clip_mlp_mapper`.
5. Save all metrics into one comparison table.
6. Add LoRA only after the four core experiments are stable.

## Team

- Hengrui Ren
