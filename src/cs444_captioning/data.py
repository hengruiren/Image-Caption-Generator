import json
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


def attach_image_paths(df, image_dir, source):
    out = df.copy()
    image_dir = Path(image_dir)
    out["image"] = out["image"].astype(str)
    out["caption"] = out["caption"].astype(str).str.strip()
    out["source"] = source
    out["image_path"] = out["image"].apply(lambda name: str(image_dir / name))
    return out[out["caption"].str.len() > 0].reset_index(drop=True)


def load_flickr8k_dataframe(data_root, source="flickr8k"):
    data_root = auto_find_data_root(data_root)
    caption_file = find_caption_file(data_root)
    image_dir = find_image_dir(data_root)
    df = load_flickr8k_captions(caption_file)
    return attach_image_paths(df, image_dir, source), image_dir, caption_file


def find_vizwiz_annotation_file(data_root, split):
    data_root = Path(data_root)
    split = split.lower()
    candidates = [
        data_root / f"{split}.json",
        data_root / f"{split}_annotations.json",
        data_root / f"VizWiz_{split}.json",
        data_root / "annotations" / f"{split}.json",
        data_root / "annotations" / f"{split}_annotations.json",
        data_root / "annotations" / f"VizWiz_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(
        path
        for path in data_root.rglob("*.json")
        if split in path.stem.lower() and "caption" in path.stem.lower()
    )
    if not matches:
        matches = sorted(path for path in data_root.rglob("*.json") if split in path.stem.lower())
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find VizWiz {split} annotation JSON under {data_root}")


def find_vizwiz_image_dir(data_root, split):
    data_root = Path(data_root)
    split = split.lower()
    preferred = [
        split,
        f"{split}_images",
        f"vizwiz_{split}",
        f"VizWiz_{split}",
        "Images",
        "images",
    ]
    for name in preferred:
        matches = [path for path in data_root.rglob(name) if path.is_dir()]
        for path in matches:
            if any(path.glob("*.jpg")) or any(path.glob("*.jpeg")) or any(path.glob("*.png")):
                return path

    prefix = f"vizwiz_{split}_"
    for path in data_root.rglob("*"):
        if path.is_dir() and any(p.name.lower().startswith(prefix) for p in path.glob("*.jpg")):
            return path

    raise FileNotFoundError(f"Could not find VizWiz {split} image directory under {data_root}")


def load_vizwiz_captions(annotation_path, image_dir, include_rejected=False, include_precanned=False):
    annotation_path = Path(annotation_path)
    image_dir = Path(image_dir)
    with open(annotation_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    image_by_id = {item["id"]: item["file_name"] for item in images}
    text_detected_by_id = {item["id"]: item.get("text_detected") for item in images}

    rows = []
    for annotation in annotations:
        if annotation.get("is_rejected", False) and not include_rejected:
            continue
        if annotation.get("is_precanned", False) and not include_precanned:
            continue

        image_id = annotation["image_id"]
        file_name = image_by_id.get(image_id)
        if file_name is None:
            continue

        caption = str(annotation.get("caption", "")).strip()
        if not caption:
            continue

        rows.append(
            {
                "image": f"vizwiz/{file_name}",
                "caption": caption,
                "source": "vizwiz",
                "image_path": str(image_dir / file_name),
                "source_image_id": image_id,
                "text_detected": text_detected_by_id.get(image_id),
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def sample_caption_images(df, max_images=None, seed=444):
    if max_images is None:
        return df.reset_index(drop=True)

    images = sorted(df["image"].unique().tolist())
    if len(images) <= max_images:
        return df.reset_index(drop=True)

    rng = random.Random(seed)
    selected = set(rng.sample(images, max_images))
    return df[df["image"].isin(selected)].reset_index(drop=True)


def load_vizwiz_split(
    data_root,
    split="train",
    max_images=None,
    seed=444,
    include_rejected=False,
    include_precanned=False,
):
    annotation_file = find_vizwiz_annotation_file(data_root, split)
    image_dir = find_vizwiz_image_dir(data_root, split)
    df = load_vizwiz_captions(
        annotation_file,
        image_dir,
        include_rejected=include_rejected,
        include_precanned=include_precanned,
    )
    return sample_caption_images(df, max_images=max_images, seed=seed), image_dir, annotation_file


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


def build_flickr8k_vizwiz_splits(
    flickr8k_root=Path("./data/flickr8k"),
    vizwiz_root=Path("./data/vizwiz"),
    vizwiz_train_images=10000,
    vizwiz_val_images=1000,
    seed=444,
):
    flickr_df, flickr_image_dir, flickr_caption_file = load_flickr8k_dataframe(flickr8k_root)
    flickr_train_df, flickr_val_df, flickr_test_df = split_by_image(flickr_df, seed=seed)

    vizwiz_train_df, vizwiz_train_image_dir, vizwiz_train_annotation = load_vizwiz_split(
        vizwiz_root,
        split="train",
        max_images=vizwiz_train_images,
        seed=seed,
    )
    vizwiz_val_df, vizwiz_val_image_dir, vizwiz_val_annotation = load_vizwiz_split(
        vizwiz_root,
        split="val",
        max_images=vizwiz_val_images,
        seed=seed,
    )

    train_df = pd.concat([flickr_train_df, vizwiz_train_df], ignore_index=True)
    val_df = pd.concat([flickr_val_df, vizwiz_val_df], ignore_index=True)
    test_df = flickr_test_df.reset_index(drop=True)
    metadata = {
        "flickr8k_image_dir": flickr_image_dir,
        "flickr8k_caption_file": flickr_caption_file,
        "vizwiz_train_image_dir": vizwiz_train_image_dir,
        "vizwiz_train_annotation": vizwiz_train_annotation,
        "vizwiz_val_image_dir": vizwiz_val_image_dir,
        "vizwiz_val_annotation": vizwiz_val_annotation,
        "vizwiz_train_images": vizwiz_train_df["image"].nunique(),
        "vizwiz_val_images": vizwiz_val_df["image"].nunique(),
    }
    return train_df, val_df, test_df, metadata


def resolve_image_path(row, image_dir):
    if "image_path" in row and pd.notna(row["image_path"]):
        return Path(row["image_path"])
    return Path(image_dir) / row["image"]


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
        image_path = resolve_image_path(row, self.image_dir)
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
