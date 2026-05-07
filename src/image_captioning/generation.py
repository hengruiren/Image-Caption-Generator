import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import COCOCaptionDataset, collate_fn


@torch.no_grad()
def generate_captions(model, val_rows, image_dir, image_processor, tokenizer, cfg, device):
    model.eval()
    seen_ids = set()
    unique_rows = []
    for r in val_rows:
        if r["image_id"] not in seen_ids:
            seen_ids.add(r["image_id"])
            unique_rows.append(r)

    dataset = COCOCaptionDataset(unique_rows, image_dir, image_processor, tokenizer, cfg.max_length)
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_fn)

    predictions = {}
    for batch in tqdm(loader, desc="Generating"):
        pixel_values = batch["pixel_values"].to(device)
        output_ids = model.generate(pixel_values, num_beams=cfg.num_beams, max_length=cfg.max_length)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for image_id, caption in zip(batch["image_id"], decoded):
            predictions[image_id] = caption.strip()

    return predictions
