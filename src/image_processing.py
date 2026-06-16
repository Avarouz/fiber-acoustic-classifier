import os
import shutil
import pandas as pd

# == directories ==

WHALE_DIR = "../data/eq-data/fin_whale"
EARTHQUAKE_DIR = "../data/eq_data/das_data/earthquakes"
NOISE_DIR = "../data/eq_data/das_data/noise_data"

OUTPUT_DIR = "test_set"

# == testing set ==

TEST_WHALES = [
    "signal_66582.h5",
    "signal_7402.h5",
    "signal_9660.h5",
    "signal_38865.h5",
    "signal_16446.h5",
]

TEST_EARTHQUAKES = [
    "ci39227895.h5",
    "ci38628799.h5",
    "ci38538991.h5",
    "ci39007775.h5",
    "ci38242914.h5",
]

TEST_NOISE = [
    "ci39812319-2.h5",
    "ci38529591.h5",
    "nn00831033.h5",
    "NCSN73972711.0.h5",
    "NCSN73774300.0.h5",
]


DATASETS = {
    "whale": (WHALE_DIR, TEST_WHALES),
    "earthquake": (EARTHQUAKE_DIR, TEST_EARTHQUAKES),
    "noise": (NOISE_DIR, TEST_NOISE),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
rows = []

for label, (source_dir, file_list) in DATASETS.items():

    class_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    for fname in file_list:

        src = os.path.join(source_dir, fname)
        dst = os.path.join(class_dir, fname)

        if not os.path.exists(src):
            print(f"[MISSING] {src}")
            continue

        shutil.copy2(src, dst)

        rows.append(
            {
                "file": fname,
                "label": label.upper(),
            }
        )

        print(f"[COPIED] {dst}")

# == save gt labels ==
labels_csv = os.path.join(OUTPUT_DIR, "test_labels.csv")

pd.DataFrame(rows).to_csv(labels_csv, index=False)

print("\n=========================")
print(f"Copied {len(rows)} files")
print(f"Saved labels to {labels_csv}")
print("=========================")