import json
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from .data import build_reference_map


def generate_caption_for_image(model, image_path, image_processor, tokenizer, cfg, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=cfg.max_length,
            num_beams=cfg.generation_num_beams,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def generate_predictions(
    model,
    eval_df,
    image_dir,
    image_processor,
    tokenizer,
    cfg,
    device,
    max_images=None,
    experiment_name="vit_no_mapper",
):
    exp_dir = cfg.output_root / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ref_map = build_reference_map(eval_df)
    image_ids = sorted(eval_df["image"].unique().tolist())
    if max_images is not None:
        image_ids = image_ids[:max_images]

    rows = []
    for image_id in tqdm(image_ids, desc=f"generate {experiment_name}"):
        pred = generate_caption_for_image(
            model,
            Path(image_dir) / image_id,
            image_processor,
            tokenizer,
            cfg,
            device,
        )
        rows.append({"image_id": image_id, "prediction": pred, "references": ref_map[image_id]})

    with open(exp_dir / "predictions.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return rows

