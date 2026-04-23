import os
import shutil
from pathlib import Path

# Directories
SOURCE_DIR_1 = Path(r"d:\065创新\garbageClassification\garbage_classification")
SOURCE_DIR_2 = Path(r"d:\065创新\archive\dataset-resized")
TARGET_DIR = Path(r"d:\065创新\unified_dataset")

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

def prepare_data():
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
