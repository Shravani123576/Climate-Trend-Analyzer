from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
TABLES_DIR = OUTPUT_DIR / "tables"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"

# Images
IMAGES_DIR = PROJECT_ROOT / "images"