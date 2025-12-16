from pathlib import Path

#Project root = folder that contains "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "validation"
TEST_DIR = PROCESSED_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
INTERFACE_DIR = PROJECT_ROOT / "interface"

# Optional: keep image size / random seed in one place later
IMG_SIZE = (224, 224)
RANDOM_SEED = 42
