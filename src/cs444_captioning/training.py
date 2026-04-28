import math
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


def make_optimizer_and_scheduler(model, train_loader, cfg):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def run_one_epoch(model, loader, device, optimizer=None, scheduler=None, cfg=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    optimizer_steps = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc=desc)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            if is_train:
                (loss / cfg.gradient_accumulation_steps).backward()

        losses.append(float(outputs.loss.detach().cpu()))

        should_step = (step + 1) % cfg.gradient_accumulation_steps == 0 or step + 1 == len(loader)
        if is_train and should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    return float(np.mean(losses)), optimizer_steps


def train_model(model, train_loader, val_loader, tokenizer, device, cfg, experiment_name):
    exp_dir = cfg.output_root / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    optimizer, scheduler = make_optimizer_and_scheduler(model, train_loader, cfg)
    history = []
    start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, _ = run_one_epoch(
            model, train_loader, device, optimizer, scheduler, cfg, desc=f"{experiment_name} epoch {epoch} train"
        )
        with torch.no_grad():
            val_loss, _ = run_one_epoch(
                model, val_loader, device, optimizer=None, scheduler=None, cfg=cfg, desc=f"{experiment_name} epoch {epoch} val"
            )

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(row)
        pd.DataFrame(history).to_csv(exp_dir / "training_log.csv", index=False)
        model.save_pretrained(exp_dir / "checkpoint_last")
        tokenizer.save_pretrained(exp_dir / "checkpoint_last")

    elapsed = time.time() - start
    print(f"Training finished in {elapsed / 60:.2f} minutes")
    return pd.DataFrame(history)

