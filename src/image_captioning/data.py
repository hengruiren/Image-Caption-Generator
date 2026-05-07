import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_coco_annotations(ann_file):
    with open(ann_file, "r") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    rows = []
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        rows.append({
            "image_id": image_id,
            "file_name": id_to_filename[image_id],
            "caption": ann["caption"].strip(),
        })
    return rows


def sample_subset(rows, max_images, image_dir=None, seed=42):
    rng = random.Random(seed)
    all_image_ids = list({r["image_id"] for r in rows})
    rng.shuffle(all_image_ids)

    # Only include images that actually exist on disk.
    # Scan the image directory directly (~30K files) instead of iterating
    # all annotation rows (~600K).  Extract image_id from the filename.
    if image_dir is not None:
        import re
        image_dir = Path(image_dir)
        existing_ids: set[int] = set()
        for fname in tqdm(list(image_dir.iterdir()), desc="Scanning image files", leave=False):
            # COCO filenames: 000000000034.jpg — just the 12-digit id
            m = re.search(r'^(\d{12})', fname.name)
            if m:
                existing_ids.add(int(m.group(1)))
        all_image_ids = [iid for iid in all_image_ids if iid in existing_ids]

    selected = set(all_image_ids[:max_images])
    return [r for r in rows if r["image_id"] in selected]


def build_reference_map(rows):
    ref_map = {}
    for r in rows:
        ref_map.setdefault(r["image_id"], []).append(r["caption"])
    return ref_map


class COCOCaptionDataset(Dataset):
    def __init__(self, rows, image_dir, image_processor, tokenizer, max_length=40, preprocess='online'):
        self.rows = rows
        self.image_dir = Path(image_dir)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess

        # Pre-resize & cache all images upfront so training only does tensor I/O.
        self._pixel_cache: dict[int, torch.Tensor] = {}
        if preprocess == 'cache':
            self._build_pixel_cache()

    def _build_pixel_cache(self):
        from tqdm import tqdm
        for row in tqdm(self.rows, desc="Preprocessing images", leave=False):
            img_path = self.image_dir / row["file_name"]
            image = Image.open(img_path).convert("RGB")
            self._pixel_cache[row["image_id"]] = (
                self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        if self.preprocess == 'cache':
            pixel_values = self._pixel_cache[row["image_id"]]
        else:
            image = Image.open(self.image_dir / row["file_name"]).convert("RGB")
            pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        tokenized = self.tokenizer(
            row["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_id": row["image_id"],
            "file_name": row["file_name"],
            "caption": row["caption"],
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "image_id": [x["image_id"] for x in batch],
        "file_name": [x["file_name"] for x in batch],
        "caption": [x["caption"] for x in batch],
    }


def build_dataloaders(cfg, image_processor, tokenizer):
    train_rows_all = load_coco_annotations(cfg.train_ann_file)
    val_rows_all   = load_coco_annotations(cfg.val_ann_file)

    train_rows = sample_subset(train_rows_all, cfg.train_subset_size, cfg.train_image_dir, cfg.seed)
    val_rows   = sample_subset(val_rows_all,   cfg.val_size,          cfg.val_image_dir,   cfg.seed)

    train_ds = COCOCaptionDataset(train_rows, cfg.train_image_dir, image_processor, tokenizer, cfg.max_length, preprocess='online')
    val_ds   = COCOCaptionDataset(val_rows,   cfg.val_image_dir,   image_processor, tokenizer, cfg.max_length, preprocess='online')

    persistent = cfg.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  num_workers=cfg.num_workers, pin_memory=True,  persistent_workers=persistent, prefetch_factor=4 if cfg.num_workers > 0 else None, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True,  persistent_workers=persistent, prefetch_factor=4 if cfg.num_workers > 0 else None, collate_fn=collate_fn)

    val_ref_map = build_reference_map(val_rows_all)

    return train_loader, val_loader, val_rows, val_ref_map


def dataset_summary(train_rows, val_rows):
    import pandas as pd
    return pd.DataFrame([
        {"split": "train", "captions": len(train_rows), "images": len({r["image_id"] for r in train_rows})},
        {"split": "val",   "captions": len(val_rows),   "images": len({r["image_id"] for r in val_rows})},
    ])
