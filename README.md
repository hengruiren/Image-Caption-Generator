# Cross-Modal Alignment in Image Captioning

**CS444 Final Project** 

## Overview

This project compares four cross-modal alignment strategies for image captioning, using a controlled encoder-decoder framework (ViT/CLIP → GPT-2). The central question is: how does the choice of visual encoder and projection module affect caption quality and cross-domain generalization?

## Research Question

In encoder-decoder multimodal architectures, how can we most effectively align the representation spaces of a pretrained vision encoder (ViT / CLIP ViT) and a pretrained language model (GPT-2)?

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

## Project Structure

```
Image_Caption_Generator/
├── src/image_captioning/
│   ├── config.py         # Hyperparameters and experiment configuration
│   ├── data.py           # COCO data loading and Dataset class
│   ├── modeling.py       # Model definitions (ViT/CLIP × No Mapper/MLP)
│   ├── training.py       # Training loop
│   ├── evaluation.py     # BLEU-4, METEOR, CIDEr evaluation
│   ├── generation.py     # Caption generation (beam search)
│   └── visualization.py  # Plots, attention heatmaps, result tables
└── image_captioning.ipynb  # Main notebook: runs all experiments and figures
```

## Setup

```bash
pip install transformers==4.44.0 datasets accelerate pycocoevalcap
pip install pandas matplotlib seaborn tqdm pillow torch
```

## Models

- **Encoder (baseline):** `google/vit-base-patch16-224-in21k` — 86M params, ImageNet-21k pretrained
- **Encoder (experiment):** `openai/clip-vit-base-patch16` — CLIP-aligned visual features
- **Decoder:** `gpt2` — 124M params; cross-attention weights trained from scratch
- **MLP Mapper:** `Linear(768→1024) → ReLU → Dropout(0.1) → Linear(1024→768) → LayerNorm`

## Training Details

- Dataset: MS-COCO 2017 (30k train subset / 5k val)
- Encoder: frozen during training
- Optimizer: AdamW, lr=5e-5
- Epochs: 5
- Batch size: 32 (A100)
- Mixed precision: FP16

## References

1. Dosovitskiy et al. (2021). An Image is Worth 16x16 Words. ICLR. (ViT)
2. Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML. (CLIP)
3. Radford et al. (2019). Language Models are Unsupervised Multitask Learners. (GPT-2)
4. Mokady et al. (2021). ClipCap: CLIP Prefix for Image Captioning. arXiv:2111.09734.
5. Li et al. (2023). BLIP-2. ICML. arXiv:2301.12597.
6. Lin, T.-Y. et al. (2014). Microsoft COCO. ECCV. (COCO dataset)
