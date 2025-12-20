from pathlib import Path

ROOT = Path("rawdataset")
splits = ["train", "valid", "test"]

for split in splits:
    labels_dir = ROOT / split / "labels"

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = [line for line in lines if line.strip().startswith("0 ") or line.strip() == "0"]

        with open(label_file, "w") as f:
            f.writelines(new_lines)