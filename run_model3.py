#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/workspace/Image_Caption_Generator/src')
os.chdir('/workspace/Image_Caption_Generator')

import torch

# Force reload the module
import importlib
import image_captioning.modeling as mod
importlib.reload(mod)

from image_captioning.modeling import build_model, load_processors
from image_captioning.training import run_training
from image_captioning.config import Config

cfg = Config()

# exp3 = experiments[2] = clip_no_mapper
exp3 = cfg.experiments[2]
print(f"Running exp3: {exp3}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

image_processor_3, tokenizer_3 = load_processors(exp3['encoder'])
print("Processors loaded")

from image_captioning.data import build_dataloaders
train_loader_3, val_loader_3, _, _ = build_dataloaders(cfg, image_processor_3, tokenizer_3)
print("Dataloaders built")

model_3 = build_model(exp3['encoder'], exp3['use_mapper'], tokenizer_3, DEVICE)
print("Model built, starting training...")

history_3 = run_training(model_3, train_loader_3, val_loader_3, cfg, exp3['name'])
print("Training complete!")
print(history_3)
