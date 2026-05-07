from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns


def plot_loss_curve(history_df, exp_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    ax.plot(history_df["epoch"], history_df["val_loss"],   marker="s", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Curve — {exp_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_results_table(results_df):
    display_df = results_df[["Experiment", "BLEU-4", "METEOR", "CIDEr"]].copy()
    display_df = display_df.sort_values("CIDEr", ascending=False).reset_index(drop=True)
    return display_df


def plot_metric_bars(results_df):
    metrics = ["BLEU-4", "METEOR", "CIDEr"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=results_df, x="Experiment", y=metric, ax=ax, palette="Blues_d")
        ax.set_title(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Experiment Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_sample_images(rows, image_dir, n=4):
    image_dir = Path(image_dir)
    seen = set()
    samples = []
    for r in rows:
        if r["image_id"] not in seen:
            seen.add(r["image_id"])
            samples.append(r)
        if len(samples) == n:
            break

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, row in zip(axes, samples):
        img = mpimg.imread(image_dir / row["file_name"])
        ax.imshow(img)
        ax.set_title(row["caption"][:60], fontsize=8, wrap=True)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_predictions_comparison(image_ids, predictions_dict, ref_map, image_dir, val_rows, n=4):
    image_dir = Path(image_dir)
    id_to_file = {r["image_id"]: r["file_name"] for r in val_rows}
    sample_ids = image_ids[:n]
    exp_names = list(predictions_dict.keys())

    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    for ax, img_id in zip(axes, sample_ids):
        img = mpimg.imread(image_dir / id_to_file[img_id])
        ax.imshow(img)
        ax.axis("off")
        lines = [f"REF: {ref_map[img_id][0]}"]
        for exp in exp_names:
            lines.append(f"{exp}: {predictions_dict[exp].get(img_id, '')}")
        ax.set_title("\n".join(lines), fontsize=8, loc="left")

    plt.tight_layout()
    plt.show()


def show_model_summary(model_infos):
    import pandas as pd
    df = pd.DataFrame(model_infos, columns=["Experiment", "Encoder", "Mapper", "Total Params", "Trainable Params"])
    return df
