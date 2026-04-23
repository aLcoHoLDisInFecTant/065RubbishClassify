import shutil
import zipfile
from pathlib import Path

# Directories (project-relative)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR_1 = DATA_DIR / "garbage_classification"
SOURCE_DIR_2 = DATA_DIR / "dataset-resized"
TARGET_DIR = BASE_DIR / "unified_dataset"

# Mappings
MAPPING = {
    "plastic": [
        SOURCE_DIR_1 / "plastic",
        SOURCE_DIR_2 / "plastic"
    ],
    "glass": [
        SOURCE_DIR_1 / "brown-glass",
        SOURCE_DIR_1 / "green-glass",
        SOURCE_DIR_1 / "white-glass",
        SOURCE_DIR_2 / "glass"
    ],
    "metal": [
        SOURCE_DIR_1 / "metal",
        SOURCE_DIR_2 / "metal"
    ],
    "paper": [
        SOURCE_DIR_1 / "cardboard",
        SOURCE_DIR_1 / "paper",
        SOURCE_DIR_2 / "cardboard",
        SOURCE_DIR_2 / "paper"
    ],
    "organic": [
        SOURCE_DIR_1 / "biological"
    ]
}

def ensure_dataset_unzipped() -> None:
    """Extract datasets in data/ if only zip files are present."""
    zip_and_target = [
        (DATA_DIR / "garbageClassification.zip", SOURCE_DIR_1),
        (DATA_DIR / "archive.zip", SOURCE_DIR_2),
    ]
    for zip_path, target_dir in zip_and_target:
        if target_dir.exists():
            continue
        if not zip_path.exists():
            print(f"Warning: missing both directory and zip for {target_dir.name}")
            continue
        print(f"Extracting {zip_path.name} to {DATA_DIR} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)

def prepare_data():
    ensure_dataset_unzipped()

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    for category, source_paths in MAPPING.items():
        cat_dir = TARGET_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        file_count = 0
        for src in source_paths:
            if src.exists():
                print(f"Processing {src} -> {category}")
                for file_path in src.glob("*.*"):
                    if file_path.is_file():
                        # Use a unique name to prevent collisions
                        dest_file = cat_dir / f"{src.name}_{file_count}_{file_path.name}"
                        shutil.copy2(file_path, dest_file)
                        file_count += 1
            else:
                print(f"Warning: {src} does not exist!")
        
        print(f"Category {category}: {file_count} files.")

if __name__ == "__main__":
    prepare_data()
    print("Data preparation complete!")
