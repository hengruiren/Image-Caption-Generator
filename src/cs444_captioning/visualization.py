from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def show_dataset_example(df, image_dir, image_name=None):
    if image_name is None:
        image_name = df.iloc[0]["image"]
    refs = df[df["image"] == image_name]["caption"].tolist()
    img = Image.open(Path(image_dir) / image_name).convert("RGB")
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    for i, cap in enumerate(refs, 1):
        print(f"{i}. {cap}")


def plot_training_history(history_df, title="Training History"):
    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def show_predictions(predictions, image_dir, n=5):
    for row in predictions[:n]:
        img = Image.open(Path(image_dir) / row["image_id"]).convert("RGB")
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        print("Image:", row["image_id"])
        print("Prediction:", row["prediction"])
        print("Reference:", row["references"][0])
        print("-" * 80)


def plot_metric_bars(results_df):
    if results_df.empty:
        print("No results to plot yet.")
        return
    metric_cols = [c for c in ["BLEU-4", "METEOR", "CIDEr"] if c in results_df.columns]
    plot_df = results_df.melt(
        id_vars=["Experiment"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=plot_df, x="Experiment", y="Score", hue="Metric")
    plt.xticks(rotation=20, ha="right")
    plt.title("Captioning Metrics")
    plt.tight_layout()
    plt.show()
