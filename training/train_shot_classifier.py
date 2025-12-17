import torch
# --- Compatibility Patch for torchvision > 0.18 ---
import sys
import torchvision
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    # pytorchvideo relies on this deprecated module
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F
# --------------------------------------------------
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    UniformTemporalSubsample,
    Normalize,
    ShortSideScale,
) 
from torchvision.transforms import Compose as VisionCompose, Lambda, RandomHorizontalFlip, RandomCrop
import os
import argparse
# --- Fix for ModuleNotFound 'engine' ---
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# ---------------------------------------
from engine.action_recognition import ShotClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def make_transform(mode="train"):
    """
    Create transform pipeline for SlowFast.
    """
    if mode == "train":
        transform = VisionCompose([
            Lambda(lambda x: x / 255.0),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            RandomShortSideScale(min_size=256, max_size=320),
            RandomCrop(256),
            RandomHorizontalFlip(p=0.5),
            UniformTemporalSubsample(32), # Target T=32
        ])
    else:
        transform = VisionCompose([
            Lambda(lambda x: x / 255.0),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ShortSideScale(size=256),
            UniformTemporalSubsample(32),
        ])
    return transform

def train_shot_classifier(data_path, output_dir, epochs=10, batch_size=32, lr=0.01):
    # Optimized for RTX 4080: Batch Size 32 (or 16 if OOM), Mixed Precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    # Assuming data_path is structured like Kinetics: train/class_name/video.mp4
    # pytorchvideo.data.Kinetics works for this structure
    
    train_transform =  ApplyTransformToKey(
        key="video",
        transform=make_transform("train"),
    )
    
    val_transform = ApplyTransformToKey(
        key="video",
        transform=make_transform("val"),
    )

    train_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(data_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", 2.0), # 2.0 second clips
        decode_audio=False,
        transform=train_transform,
    )
    
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(data_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", 2.0),
        decode_audio=False,
        transform=val_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    # Initialize Model
    # Note: Ensure the num_classes matches the dataset folders
    # We can count classes from the directory
    train_dir = os.path.join(data_path, 'train')
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    
    classifier = ShotClassifier(num_classes=num_classes, device=device)
    model = classifier.model
    
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            # Inputs: list of [slow, fast]
            video = batch['video'].to(device) # Shape (B, C, T, H, W)
            labels = batch['label'].to(device)

            # Pack for SlowFast
            inputs = classifier._pack_pathway_output(video)
            
            optimizer.zero_grad()
            
            # AMP Context
            with torch.cuda.amp.autocast():
                preds = model(inputs)
                loss = criterion(preds, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_batches = 0
        with torch.no_grad():
            # Validation loop with progress bar
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                video = batch['video'].to(device)
                labels = batch['label'].to(device)
                
                inputs = classifier._pack_pathway_output(video)
                preds = model(inputs)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_batches += 1
                
                _, predicted = preds.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"shot_classifier_epoch_{epoch+1}.pth"))
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        acc = 100.*correct/total if total > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.2f}%")
        scheduler.step()
        
        
    print("Training Complete. Final model saved.")
    
    # --- Final Evaluation with Detailed Metrics ---
    print("\nRunning Final Evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
             video = batch['video'].to(device)
             labels = batch['label'].cpu().numpy()
             
             inputs = classifier._pack_pathway_output(video)
             preds = model(inputs)
             _, predicted = preds.max(1)
             
             all_preds.extend(predicted.cpu().numpy())
             all_labels.extend(labels)
             
    # Classification Report
    # Sort classes locally for consistent reporting (assuming folder sort)
    sorted_classes = sorted(classes)
    
    report = classification_report(all_labels, all_preds, target_names=sorted_classes)
    print("\nClassification Report:\n")
    print(report)
    
    # Save report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        cm = confusion_matrix(all_labels, all_preds)
        start_idx = 0
        for i, row in enumerate(cm):
             f.write(f"{sorted_classes[i]}: {row}\n")

    print(f"Evaluation report saved to {output_dir}/classification_report.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset root (containing train/ and val/ folders)")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (32 recommended for RTX 4080)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_shot_classifier(args.data_path, args.output_dir, args.epochs, args.batch_size)
