# Cross-Modal Alignment in Image Captioning

## Overview

This project investigates cross-modal alignment strategies for image captioning within a controlled encoder-decoder framework (ViT/CLIP → GPT-2). Four configurations are compared by varying the visual encoder (Vanilla ViT vs. CLIP ViT) and whether an explicit MLP projection module bridges the two modalities. All models are trained and evaluated on MS-COCO 2017 to isolate the effect of each alignment choice on caption quality and cross-domain generalization.

## Research Question

In encoder-decoder multimodal architectures, how does the choice of visual encoder (Vanilla ViT vs. CLIP ViT) and projection module (no mapper vs. MLP mapper) affect caption quality and cross-domain generalization?

## Experiment Matrix

| Experiment | Encoder | Mapper | Description |
|---|---|---|---|
| `vit_no_mapper` | Vanilla ViT | None | Baseline: implicit alignment via cross-attention |
| `vit_mlp_mapper` | Vanilla ViT | MLP | Explicit projection from visual to language space |
| `clip_no_mapper` | CLIP ViT | None | Pre-aligned visual features, no explicit bridge |
| `clip_mlp_mapper` | CLIP ViT | MLP | Pre-aligned features + explicit projection |

All models are trained on a 30k subset of MS-COCO train2017 and evaluated on COCO val2017 (5k images).

## Metrics

- **BLEU-4** — 4-gram precision between generated and reference captions
- **METEOR** — Harmonic mean of precision and recall, accounts for synonyms
- **CIDEr** — Consensus-based similarity, weighted by term relevance

## Datasets

This project uses **MS-COCO 2017**. The `datasets` library can download it automatically when you run the notebook.

**Automatic (recommended):** Handled via HuggingFace `datasets`:
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceM4/COCO", trust_remote_code=True)
```

**Manual download:**
1. Visit [https://cocodataset.org/#download](https://cocodataset.org/#download)
2. Download:
   - `2017 Train images` (~18 GB)
   - `2017 Val images` (~1 GB)
   - `2017 Train/Val annotations`
3. Extract into a local directory and update the data path in `src/image_captioning/config.py`

## Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install transformers==4.44.0 accelerate datasets peft
pip install pycocoevalcap pycocotools
pip install pandas numpy matplotlib seaborn tqdm pillow scikit-learn nltk
```

### 2. Run via notebook

Open and run all cells in `image_captioning.ipynb`. The notebook:
- Downloads and preprocesses the COCO dataset
- Trains all four model configurations sequentially
- Evaluates with BLEU-4, METEOR, and CIDEr
- Generates visualizations and comparison tables

```bash
jupyter notebook image_captioning.ipynb
```

### 3. Run model 3 standalone

To train and evaluate the CLIP + MLP mapper configuration independently:

```bash
python run_model3.py
```

## Project Structure

```
Image_Caption_Generator/
├── src/
│   └── image_captioning/
│       ├── config.py         # Hyperparameters and experiment configuration
│       ├── data.py           # COCO data loading and Dataset class
│       ├── modeling.py       # Model definitions (ViT/CLIP × No Mapper/MLP)
│       ├── training.py       # Training loop
│       ├── evaluation.py     # BLEU-4, METEOR, CIDEr evaluation
│       ├── generation.py     # Caption generation (beam search)
│       └── visualization.py  # Plots, attention heatmaps, result tables
├── image_captioning.ipynb    # Main notebook: runs all experiments and figures
├── run_model3.py             # Standalone script for CLIP + MLP mapper
└── requirements.txt
```

## Models

| Component | Model | Parameters | Notes |
|---|---|---|---|
| Encoder (baseline) | `google/vit-base-patch16-224-in21k` | 86M | ImageNet-21k pretrained |
| Encoder (CLIP) | `openai/clip-vit-base-patch16` | 86M | CLIP-aligned visual features |
| Decoder | `gpt2` | 124M | Cross-attention weights trained from scratch |
| MLP Mapper | Linear(768→1024) → ReLU → Dropout(0.1) → Linear(1024→768) → LayerNorm | — | Optional visual-to-language projection |

Encoders are **frozen** during training; only the decoder cross-attention and optional mapper are updated.

## Training Details

| Setting | Value |
|---|---|
| Dataset | MS-COCO 2017 |
| Train split | 30k subset of train2017 |
| Val split | val2017 (5k images) |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Epochs | 5 |
| Batch size | 32 |
| Mixed precision | FP16 |
| Encoder | Frozen |

## References

1. Dosovitskiy et al. (2021). An Image is Worth 16x16 Words. ICLR. (ViT)
2. Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML. (CLIP)
3. Radford et al. (2019). Language Models are Unsupervised Multitask Learners. (GPT-2)
4. Mokady et al. (2021). ClipCap: CLIP Prefix for Image Captioning. arXiv:2111.09734.
5. Li et al. (2023). BLIP-2. ICML. arXiv:2301.12597.
6. Lin, T.-Y. et al. (2014). Microsoft COCO. ECCV. (COCO dataset)
