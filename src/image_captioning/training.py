from pathlib import Path

import torch
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, fp16):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        if fp16:
            with autocast():
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(pixel_values=pixel_values, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, device, fp16):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        if fp16:
            with autocast():
                out = model(pixel_values=pixel_values, labels=labels)
        else:
            out = model(pixel_values=pixel_values, labels=labels)
        total_loss += out.loss.item()
    return total_loss / len(loader)


def run_training(model, train_loader, val_loader, cfg, exp_name):
    device = next(model.parameters()).device
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    history = []
    best_val_loss = float("inf")
    ckpt_dir = cfg.output_root / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg.fp16)
        val_loss = evaluate_loss(model, val_loader, device, cfg.fp16)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{exp_name}] Epoch {epoch}/{cfg.num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    return pd.DataFrame(history)
