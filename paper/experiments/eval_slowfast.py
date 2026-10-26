"""Re-evaluate the fine-tuned SlowFast shot classifier on the VideoBadminton validation split.

Writes paper/results/slowfast_metrics.json, slowfast_logits.npz and the confusion/F1 figures.
No training is performed. Usage: python paper/experiments/eval_slowfast.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from engine.action_recognition import ShotClassifier  # applies torchvision compat patch first

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorchvideo.data
from pytorchvideo.transforms import ApplyTransformToKey
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from train_shot_classifier import make_transform

DATA = os.path.join(ROOT, "data", "VideoBadminton_Split")
OUT = os.path.join(ROOT, "paper", "results")
FIG = os.path.join(ROOT, "paper", "figures")
rng = np.random.default_rng(0)


def bootstrap_ci(y, p, fn, n=2000):
    idx = np.arange(len(y))
    vals = [fn(y[s], p[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    classes = sorted(d for d in os.listdir(os.path.join(DATA, "val")) if os.path.isdir(os.path.join(DATA, "val", d)))
    clf = ShotClassifier(num_classes=len(classes), device="cuda")
    ds = pytorchvideo.data.Kinetics(
        data_path=os.path.join(DATA, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", 2.0),
        decode_audio=False,
        transform=ApplyTransformToKey(key="video", transform=make_transform("val")),
    )
    loader = DataLoader(ds, batch_size=16, num_workers=4)

    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].cuda()
            logits.append(clf.model(clf._pack_pathway_output(video)).float().cpu())
            labels.append(batch["label"])
    logits = torch.cat(logits)
    y = torch.cat(labels).numpy()
    p = logits.argmax(1).numpy()
    top5 = float((logits.topk(5, dim=1).indices.numpy() == y[:, None]).any(1).mean())

    cm = confusion_matrix(y, p, labels=range(len(classes)))
    per_class_acc = cm.diagonal() / cm.sum(1)
    macro_f1 = lambda a, b: f1_score(a, b, average="macro")
    acc_fn = lambda a, b: float((a == b).mean())

    # Latency: single 32-frame clip, fp32, GPU, after warm-up
    sample = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))["video"].cuda()
    inp = clf._pack_pathway_output(sample)
    with torch.no_grad():
        for _ in range(10):
            clf.model(list(inp))
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            clf.model(list(inp))
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 100 * 1000

    metrics = {
        "n_clips": int(len(y)),
        "n_classes": len(classes),
        "top1_acc": acc_fn(y, p),
        "top1_acc_ci95": bootstrap_ci(y, p, acc_fn),
        "top5_acc": top5,
        "macro_f1": float(macro_f1(y, p)),
        "macro_f1_ci95": bootstrap_ci(y, p, macro_f1),
        "weighted_f1": float(f1_score(y, p, average="weighted")),
        "mean_class_acc": float(per_class_acc.mean()),
        "latency_ms_per_clip_bs1": latency_ms,
        "gpu": torch.cuda.get_device_name(0),
        "classes": classes,
        "per_class": classification_report(y, p, target_names=classes, output_dict=True),
    }
    json.dump(metrics, open(os.path.join(OUT, "slowfast_metrics.json"), "w"), indent=2)
    np.savez(os.path.join(OUT, "slowfast_logits.npz"), logits=logits.numpy(), labels=y)
    print({k: v for k, v in metrics.items() if k not in ("classes", "per_class")})

    short = [c.split("_", 1)[1] for c in classes]
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(short)), short, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(short)), short, fontsize=8)
    for i in range(len(short)):
        for j in range(len(short)):
            if cmn[i, j] >= 0.05:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confusion_matrix.png"), dpi=200)

    f1 = [metrics["per_class"][c]["f1-score"] for c in classes]
    sup = [metrics["per_class"][c]["support"] for c in classes]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([short[i] for i in order], [f1[i] for i in order], color="#4c8f00")
    for r, i in enumerate(order):
        ax.text(f1[i] + 0.01, r, f"n={sup[i]}", va="center", fontsize=7)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1-score")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "per_class_f1.png"), dpi=200)


if __name__ == "__main__":
    main()
