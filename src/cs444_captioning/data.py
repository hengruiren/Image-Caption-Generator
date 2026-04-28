import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def auto_find_data_root(default_root=Path("./data/flickr8k")):
    candidates = [
        Path("/kaggle/input/flickr8k"),
        Path("/kaggle/input/flickr8k-dataset"),
        Path("./data/flickr8k"),
        Path("./flickr8k"),
        Path.cwd(),
    ]
    for root in candidates:
        if root.exists():
            caption_files = list(root.rglob("captions.txt")) + list(root.rglob("Flickr8k.token.txt"))
            image_dirs = [
                p
                for p in root.rglob("*")
                if p.is_dir() and p.name.lower() in {"images", "flicker8k_dataset", "flickr8k_dataset"}
            ]
            if caption_files and image_dirs:
                return root
    return Path(default_root)


def find_caption_file(data_root):
    data_root = Path(data_root)
    for name in ["captions.txt", "Flickr8k.token.txt"]:
        matches = list(data_root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find captions.txt or Flickr8k.token.txt under {data_root}")


def find_image_dir(data_root):
    data_root = Path(data_root)
    preferred = ["Images", "images", "Flicker8k_Dataset", "Flickr8k_Dataset"]
    for name in preferred:
        matches = [p for p in data_root.rglob(name) if p.is_dir()]
        for path in matches:
            if any(path.glob("*.jpg")) or any(path.glob("*.jpeg")) or any(path.glob("*.png")):
                return path

    for path in data_root.rglob("*"):
        if path.is_dir():
            image_count = (
                sum(1 for _ in path.glob("*.jpg"))
                + sum(1 for _ in path.glob("*.jpeg"))
                + sum(1 for _ in path.glob("*.png"))
            )
            if image_count > 1000:
                return path
    raise FileNotFoundError(f"Could not find image directory under {data_root}")


def load_flickr8k_captions(caption_path):
    caption_path = Path(caption_path)
    if caption_path.name == "captions.txt":
        df = pd.read_csv(caption_path)
        df.columns = [c.strip().lower() for c in df.columns]
        image_col = "image" if "image" in df.columns else df.columns[0]
        caption_col = "caption" if "caption" in df.columns else df.columns[1]
        out = df[[image_col, caption_col]].rename(columns={image_col: "image", caption_col: "caption"})
    else:
        rows = []
        with open(caption_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                image_part, caption = line.split("\t", 1)
                rows.append({"image": image_part.split("#", 1)[0], "caption": caption})
        out = pd.DataFrame(rows)

    out["image"] = out["image"].astype(str)
    out["caption"] = out["caption"].astype(str).str.strip()
    return out[out["caption"].str.len() > 0].reset_index(drop=True)


def split_by_image(df, train_ratio=0.8, val_ratio=0.1, seed=444):
    images = sorted(df["image"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(images)

    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * val_ratio)
    train_images = set(images[:n_train])
    val_images = set(images[n_train : n_train + n_val])
    test_images = set(images[n_train + n_val :])

    train_df = df[df["image"].isin(train_images)].reset_index(drop=True)
    val_df = df[df["image"].isin(val_images)].reset_index(drop=True)
    test_df = df[df["image"].isin(test_images)].reset_index(drop=True)
    return train_df, val_df, test_df


class Flickr8kCaptionDataset(Dataset):
    def __init__(self, df, image_dir, image_processor, tokenizer, max_length=40):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row["image"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        tokenized = self.tokenizer(
            row["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_id": row["image"],
            "caption": row["caption"],
        }


def collate_caption_batch(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "image_id": [x["image_id"] for x in batch],
        "caption": [x["caption"] for x in batch],
    }


def build_reference_map(df):
    return df.groupby("image")["caption"].apply(list).to_dict()

