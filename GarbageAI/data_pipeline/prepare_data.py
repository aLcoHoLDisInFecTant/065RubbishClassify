import os
import shutil
from pathlib import Path

# Input directories
GARBAGE_CLASSIFICATION_DIR = Path(r"d:\065创新\garbageClassification\garbage_classification")
ARCHIVE_DIR = Path(r"d:\065创新\archive\dataset-resized")

# Output directory
OUTPUT_DIR = Path(r"d:\065创新\GarbageAI\dataset")

# Mapping to 5 categories
# plastic, glass, metal, paper, organic
CATEGORY_MAPPING = {
    "plastic": "plastic",
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    "glass": "glass",
    "metal": "metal",
    "battery": "metal",
    "cardboard": "paper",
    "paper": "paper",
    "biological": "organic",
}

def setup_output_dirs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    for category in set(CATEGORY_MAPPING.values()):
        (OUTPUT_DIR / category).mkdir(parents=True, exist_ok=True)
        
def process_directory(source_dir, prefix):
    if not source_dir.exists():
        print(f"Directory not found: {source_dir}")
        return

    print(f"Processing directory: {source_dir}")
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        if class_name not in CATEGORY_MAPPING:
            print(f"Skipping unmapped class: {class_name}")
            continue
            
        target_category = CATEGORY_MAPPING[class_name]
        target_dir = OUTPUT_DIR / target_category
        
        # Copy files
        count = 0
        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                new_filename = f"{prefix}_{class_name}_{file_path.name}"
                shutil.copy2(file_path, target_dir / new_filename)
                count += 1
                
        print(f"Copied {count} files from {class_name} to {target_category}")

def main():
    print("Starting data preparation...")
    setup_output_dirs()
    
    process_directory(GARBAGE_CLASSIFICATION_DIR, "ds1")
    process_directory(ARCHIVE_DIR, "ds2")
    
    print("Data preparation completed.")
    
    # Print summary
    print("\nDataset Summary:")
    for category in set(CATEGORY_MAPPING.values()):
        category_dir = OUTPUT_DIR / category
        count = len(list(category_dir.glob("*.*")))
        print(f"{category}: {count} images")

if __name__ == "__main__":
    main()
