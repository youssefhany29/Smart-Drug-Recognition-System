import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError  

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
)
SUPPORTED_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".webp"]

@dataclass
class SplitConfig:
    train: float
    val: float
    test: float
    
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare dataset: merge, shuffle, split, rename, and summarize."        
    )
    
    p.add_argument("--train", type=float, default=0.70, help="Train ratio (default: 0.70)")
    p.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    p.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")

    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default from config)")

    p.add_argument(
        "--exts",
        type=str,
        default=",".join(SUPPORTED_EXTS_DEFAULT),
        help="Comma-separated allowed extensions (default: .jpg,.jpeg,.png,.webp)",
    )
    
    p.add_argument(
        "--clean-processed",
        action="store_true",
        help="If set, deletes dataset/processed before generating new splits.",
    )
    
    p.add_argument(
        "--skip-corrupt",
        action="store_true",
        help="If set, skip unreadable/corrupt images (recommended).",
    )
    
    return p.parse_args()

def validate_splits(cfg: SplitConfig) -> None:
    total = cfg.train + cfg.val + cfg.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0. Got {total:.4f}")
    
    for name, v in [("train", cfg.train), ("val", cfg.val), ("test", cfg.test)]:
        if v <= 0:
            raise ValueError(f"{name} ratio must be > 0. Got {v}")
        

def ensure_dirs() -> None:
    for base in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (base / "original").mkdir(parents=True, exist_ok=True)
        (base / "counterfeit").mkdir(parents=True, exist_ok=True)
        
    (RESULTS_DIR / "dataset_summary").mkdir(parents=True, exist_ok=True)
    
    
def clean_processed_dir() -> None:
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    ensure_dirs()
    

def list_images(folder: Path, exts: List[str]) -> List[Path]:
    if not folder.exists():
        return []
    

    exts_lower = {e.lower().strip() for e in exts}
        
    files = []

    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            files.append(p)

    return sorted(files)

def is_image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def split_list(items: List[Path], cfg: SplitConfig) -> Tuple[List[Path], List[Path], List[Path]]:
    n = len(items)
    n_train = int(n * cfg.train)
    n_val = int(n * cfg.val)
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:n_train + n_val + n_test]
    return train_items, val_items, test_items


def copy_and_rename(
    src_files: List[Path],
    dst_dir: Path,
    prefix: str,
    start_index: int = 1,
) -> int:
    """
    Copies files into dst_dir with standardized names like:
    original_000001.jpg  or  counterfeit_000001.png
    Returns next index after finishing.
    """
    idx = start_index
    for f in src_files:
        new_name = f"{prefix}_{idx:06d}{f.suffix.lower()}"  
        dst_path = dst_dir / new_name
        shutil.copy2(f, dst_path)  
        idx += 1
    return idx

def main() -> None:
    args = parse_args()
    cfg = SplitConfig(train=args.train, val=args.val, test=args.test)
    validate_splits(cfg)
    
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    random.seed(args.seed) 
    
    raw_original = RAW_DIR / "original"
    raw_counterfeit = RAW_DIR / "counterfeit"

    if args.clean_processed:
        clean_processed_dir()
    else:
        ensure_dirs()
        
    original_files = list_images(raw_original, exts)
    counterfeit_files = list_images(raw_counterfeit, exts)
    
    if len(original_files) == 0 and len(counterfeit_files) == 0:
        raise RuntimeError(
            f"No images found. Check your raw dataset paths:\n"
            f"- {raw_original}\n- {raw_counterfeit}\n"
            f"and allowed extensions: {exts}"
        )
        
    skipped_corrupt = {"original": 0, "counterfeit": 0}
    if args.skip_corrupt:
        original_ok = []
        for p in original_files:
            if is_image_readable(p):
                original_ok.append(p)
            else:
                skipped_corrupt["original"] += 1
        original_files = original_ok

        counterfeit_ok = []
        for p in counterfeit_files:
            if is_image_readable(p):
                counterfeit_ok.append(p)
            else:
                skipped_corrupt["counterfeit"] += 1
        counterfeit_files = counterfeit_ok
        
    random.shuffle(original_files)
    random.shuffle(counterfeit_files)
    
    orig_train, orig_val, orig_test = split_list(original_files, cfg)
    fake_train, fake_val, fake_test = split_list(counterfeit_files, cfg)
    
    copy_and_rename(orig_train, TRAIN_DIR / "original", prefix="original", start_index=1)
    copy_and_rename(fake_train, TRAIN_DIR / "counterfeit", prefix="counterfeit", start_index=1)

    copy_and_rename(orig_val, VAL_DIR / "original", prefix="original", start_index=1)
    copy_and_rename(fake_val, VAL_DIR / "counterfeit", prefix="counterfeit", start_index=1)

    copy_and_rename(orig_test, TEST_DIR / "original", prefix="original", start_index=1)
    copy_and_rename(fake_test, TEST_DIR / "counterfeit", prefix="counterfeit", start_index=1)
    
    summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "ratios": {"train": cfg.train, "validation": cfg.val, "test": cfg.test},
        "extensions": exts,
        "raw_counts": {
            "original": len(original_files),
            "counterfeit": len(counterfeit_files),
            "total": len(original_files) + len(counterfeit_files),
        },
        "processed_counts": {
            "train": {"original": len(orig_train), "counterfeit": len(fake_train), "total": len(orig_train) + len(fake_train)},
            "validation": {"original": len(orig_val), "counterfeit": len(fake_val), "total": len(orig_val) + len(fake_val)},
            "test": {"original": len(orig_test), "counterfeit": len(fake_test), "total": len(orig_test) + len(fake_test)},
        },
        "skipped_corrupt": skipped_corrupt if args.skip_corrupt else None,
        "paths": {
            "raw_original": str(raw_original),
            "raw_counterfeit": str(raw_counterfeit),
            "processed": str(PROCESSED_DIR),
        },
        "note": "Each image is placed in exactly one split to avoid data leakage.",
    }
    
    out_dir = RESULTS_DIR / "dataset_summary"
    out_json = out_dir / "summary.json"
    out_txt = out_dir / "summary.txt"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    txt_lines = [
        "=== Smart Drug Recognition: Dataset Summary ===",
        f"Created at (UTC): {summary['created_at']}",
        f"Seed: {summary['seed']}",
        f"Ratios: train={cfg.train}, validation={cfg.val}, test={cfg.test}",
        f"Allowed exts: {', '.join(exts)}",
        "",
        f"RAW counts: original={summary['raw_counts']['original']}, counterfeit={summary['raw_counts']['counterfeit']}, total={summary['raw_counts']['total']}",
        "",
        "PROCESSED counts:",
        f"  Train      -> original={len(orig_train)}, counterfeit={len(fake_train)}, total={len(orig_train) + len(fake_train)}",
        f"  Validation -> original={len(orig_val)}, counterfeit={len(fake_val)}, total={len(orig_val) + len(fake_val)}",
        f"  Test       -> original={len(orig_test)}, counterfeit={len(fake_test)}, total={len(orig_test) + len(fake_test)}",
    ]

    if args.skip_corrupt:
        txt_lines += [
            "",
            f"Skipped corrupt/unreadable images: original={skipped_corrupt['original']}, counterfeit={skipped_corrupt['counterfeit']}",
        ]

    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    print("\n".join(txt_lines))
    print(f"\nSaved summary to:\n- {out_json}\n- {out_txt}")
    print("\nDONE Processed dataset is ready.")


if __name__ == "__main__":
    main()