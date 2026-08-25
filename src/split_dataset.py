from pathlib import Path
import shutil
import re

# ==========================================================
# TERRASPECTRA - SOURCE FILE LEVEL DATASET SPLIT
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

# Classes
CLASSES = [
    "alternaria_alternata",
    "alternaria_solani",
    "botrytis_cinerea",
    "fusarium_oxysporum",
    "healthy"
]


def get_file_number(file_path):
    """
    Extract sample number from filenames such as:
    Alternaria_alternata_1_pca30.npy -> 1
    Healthy_8_pca30.npy -> 8
    """

    match = re.search(r"_(\d+)_pca30\.npy$", file_path.name)

    if match:
        return int(match.group(1))

    return None


def main():

    print("=" * 60)
    print("TERRASPECTRA - DATASET SPLITTING")
    print("=" * 60)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Processed data folder not found:\n{INPUT_DIR}"
        )

    # Create train / validation / test folders
    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            folder = OUTPUT_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)

    summary = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    # Process every class
    for class_name in CLASSES:

        print("\n" + "=" * 60)
        print(f"CLASS: {class_name.upper()}")
        print("=" * 60)

        input_class_dir = INPUT_DIR / class_name

        files = sorted(input_class_dir.glob("*.npy"))

        print(f"Files found: {len(files)}")

        for file_path in files:

            file_number = get_file_number(file_path)

            if file_number is None:
                print(f"WARNING: Cannot identify number: {file_path.name}")
                continue

            # Decide dataset split
            if 1 <= file_number <= 6:
                split = "train"

            elif file_number == 7:
                split = "val"

            elif file_number == 8:
                split = "test"

            else:
                print(f"WARNING: Unexpected file: {file_path.name}")
                continue

            destination = (
                OUTPUT_DIR /
                split /
                class_name /
                file_path.name
            )

            # Copy file instead of moving it
            shutil.copy2(file_path, destination)

            summary[split] += 1

            print(
                f"{file_path.name} "
                f"-> {split.upper()}"
            )

    # Final summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)

    print(f"Train files: {summary['train']}")
    print(f"Validation files: {summary['val']}")
    print(f"Test files: {summary['test']}")

    print("\nExpected:")
    print("Train:      30 files")
    print("Validation: 5 files")
    print("Test:       5 files")

    print("\nDataset splitting completed successfully.")


if __name__ == "__main__":
    main()