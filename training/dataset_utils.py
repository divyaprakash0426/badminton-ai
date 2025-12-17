import os
import shutil
import random
import glob
import argparse
from pathlib import Path
from collections import Counter

class BadmintonDatasetManager:
    """
    Manager for Badminton Datasets (VideoBadminton, FineBadminton, etc.)
    Handles preparation, splitting, and statistics.
    """
    
    def __init__(self, root_dir="data"):
        self.root_dir = root_dir
        
    def info_videobadminton(self):
        print("===================================================")
        print("VideoBadminton Dataset Info")
        print("===================================================")
        print("Source: https://github.com/Rhinebrother/VideoBadminton")
        print("Paper: https://arxiv.org/abs/2403.12385")
        print("Classes: 18 (Short Serve, Smash, Clear, etc.)")
        print("Structure: Folders by Class Name (e.g., '00_Short Serve')")
        print("Action: Use 'split' to prepare for training.")
        print("===================================================")

    def info_shuttleset(self):
        print("===================================================")
        print("ShuttleSet Info")
        print("===================================================")
        print("Focus: Tactical Analysis (Stroke sequences)")
        print("Source: https://github.com/Poddar-Ankur/ShuttleSet")
        print("===================================================")

    def split_dataset(self, source_dir, output_dir, val_ratio=0.2, seed=42):
        """
        Splits a folder-based dataset into train/val.
        
        Args:
            source_dir (str): Path containing class folders (e.g. data/VideoBadminton).
            output_dir (str): Destination path (e.g. data/VideoBadminton_Split).
            val_ratio (float): Fraction of data for validation.
        """
        random.seed(seed)
        src_path = Path(source_dir)
        dest_path = Path(output_dir)
        
        if not src_path.exists():
            print(f"Error: Source directory {src_path} does not exist.")
            return

        # Identify classes (subdirectories)
        classes = [d for d in src_path.iterdir() if d.is_dir()]
        classes.sort()
        
        if not classes:
            print("No class folders found in source directory.")
            return

        print(f"Found {len(classes)} classes.")
        
        stats = {}
        
        for cls_dir in classes:
            cls_name = cls_dir.name
            
            # Gather video files
            extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
            files = []
            for ext in extensions:
                files.extend(list(cls_dir.glob(ext)))
            
            # Shuffle
            random.shuffle(files)
            
            # Split
            num_val = int(len(files) * val_ratio)
            val_files = files[:num_val]
            train_files = files[num_val:]
            
            stats[cls_name] = {'train': len(train_files), 'val': len(val_files)}
            
            # Copy
            for split, file_list in [('train', train_files), ('val', val_files)]:
                split_dir = dest_path / split / cls_name
                split_dir.mkdir(parents=True, exist_ok=True)
                
                for f in file_list:
                    shutil.copy2(f, split_dir / f.name)
        
        print(f"Processing complete. Output at: {dest_path}")
        self._print_stats(stats)
        
    def check_balance(self, dataset_dir):
        """
        Analyzes the class distribution of an existing dataset.
        """
        path = Path(dataset_dir)
        if not path.exists():
            print(f"Path {path} not found.")
            return

        print(f"\nDistrubution for: {dataset_dir}")
        print(f"{'Class':<30} | {'Count':<10}")
        print("-" * 45)
        
        total = 0
        counts = {}
        
        # Check if it's a split root (train/val) or flat
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if 'train' in [d.name for d in subdirs]:
             # It's a split dataset, recurse
             for split in ['train', 'val']:
                 print(f"--- {split.upper()} ---")
                 self.check_balance(path / split)
             return

        for cls_dir in subdirs:
            cls_count = len(list(cls_dir.glob('*.*')))
            counts[cls_dir.name] = cls_count
            total += cls_count
            print(f"{cls_dir.name:<30} | {cls_count:<10}")
            
        print("-" * 45)
        print(f"{'TOTAL':<30} | {total:<10}")

    def _print_stats(self, stats):
        print("\nSplit Statistics:")
        print(f"{'Class':<30} | {'Train':<8} | {'Val':<8} | {'Total':<8}")
        print("-" * 60)
        total_train = 0
        total_val = 0
        
        for cls, counts in stats.items():
            t = counts['train']
            v = counts['val']
            tot = t + v
            total_train += t
            total_val += v
            print(f"{cls:<30} | {t:<8} | {v:<8} | {tot:<8}")
            
        print("-" * 60)
        print(f"{'TOTAL':<30} | {total_train:<8} | {total_val:<8} | {total_train+total_val:<8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Badminton Dataset Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Info Command
    parser_info = subparsers.add_parser("info", help="Show dataset info")
    parser_info.add_argument("--dataset", choices=["videobadminton", "shuttleset"], default="videobadminton")
    
    # Split Command
    parser_split = subparsers.add_parser("split", help="Split dataset into train/val")
    parser_split.add_argument("--src", required=True, help="Source directory with class folders")
    parser_split.add_argument("--dest", required=True, help="Destination directory")
    parser_split.add_argument("--ratio", type=float, default=0.2, help="Validation ratio (default 0.2)")
    
    # Balance Command
    parser_bal = subparsers.add_parser("balance", help="Check class balance")
    parser_bal.add_argument("--path", required=True, help="Dataset directory to analyze")
    
    args = parser.parse_args()
    
    manager = BadmintonDatasetManager()
    
    if args.command == "info":
        if args.dataset == "videobadminton":
            manager.info_videobadminton()
        else:
            manager.info_shuttleset()
            
    elif args.command == "split":
        manager.split_dataset(args.src, args.dest, args.ratio)
        
    elif args.command == "balance":
        manager.check_balance(args.path)
        
    else:
        parser.print_help()
