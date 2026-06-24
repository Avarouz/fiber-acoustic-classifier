import os
import random
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend


# ===== directories =====

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "test_fixed"
)

PNG_DIR = os.path.join(
    OUTPUT_DIR,
    "pngs"
)

os.makedirs(PNG_DIR, exist_ok=True)


# ===== source data =====

WHALE_DIR = "../data/eq_data/fin_whale"
EARTHQUAKE_DIR = "../data/eq_data/das_data/earthquakes"
NOISE_DIR = "../data/eq_data/das_data/noise_data"

N_PER_CLASS = 10
RANDOM_SEED = 42

EXCLUDE_FILES = {
    # whales
    "signal_66582.h5",
    "signal_1131.h5",
    "signal_2125.h5",

    # earthquakes
    "ci39007775.h5",
    "ci38971232.h5",
    "nn00832186-1.h5",

    # noise
    "ci39812319-2.h5",
    "ci39281440.h5",
    "ci38538991.h5",
}

# same preprocessing for ALL classes
LOWCUT = 2.0
HIGHCUT = 10.0
FS = 100.0


# ===== helpers =====

def find_first_dataset(group):
    for key in group:
        obj = group[key]

        if isinstance(obj, h5py.Dataset):
            return obj[:]

        elif isinstance(obj, h5py.Group):
            result = find_first_dataset(obj)

            if result is not None:
                return result

    return None


def load_h5_data(filepath):
    with h5py.File(filepath, "r") as f:

        if "data" in f:
            data = f["data"][:]
        else:
            data = find_first_dataset(f)

    if data is None:
        raise ValueError(
            f"No dataset found in {filepath}"
        )

    return data


def bandpass(data,
             lowcut=LOWCUT,
             highcut=HIGHCUT,
             fs=FS,
             order=4):

    sos = butter(
        order,
        [lowcut, highcut],
        btype="band",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        data,
        axis=1
    )


def save_png(data, output_file):
    vmax = np.percentile(
        np.abs(data),
        99
    )

    plt.figure(figsize=(12, 6))

    plt.imshow(
        data,
        aspect="auto",
        cmap="seismic",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
    )

    plt.axis("off")

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()


def process_file(filepath,
                 label,
                 png_dir):

    data = load_h5_data(filepath)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    data = detrend(
        data,
        axis=1
    )

    # ==== SAME FILTER FOR EVERYTHING ====
    data = bandpass(data)

    base = os.path.splitext(
        os.path.basename(filepath)
    )[0]

    png_file = os.path.join(
        png_dir,
        f"{label}_{base}.png"
    )

    save_png(
        data,
        png_file
    )

    return os.path.basename(png_file)


def get_random_files(directory, n):
    files = [
        f
        for f in os.listdir(directory)
        if f.endswith(".h5")
        and f not in EXCLUDE_FILES
    ]

    return random.sample(
        files,
        min(n, len(files))
    )


# ===== main =====

def main():

    random.seed(RANDOM_SEED)

    datasets = {
        "whale": WHALE_DIR,
        "earthquake": EARTHQUAKE_DIR,
        "noise": NOISE_DIR,
    }

    rows = []

    for label, source_dir in datasets.items():
        selected = get_random_files(
            source_dir,
            N_PER_CLASS
        )

        print(
            f"{label}: selected {len(selected)} files"
        )

        for fname in selected:
            filepath = os.path.join(
                source_dir,
                fname
            )

            try:

                png_name = process_file(
                    filepath,
                    label,
                    PNG_DIR
                )

                rows.append(
                    {
                        "file": png_name,
                        "label": label.upper()
                    }
                )

                print(
                    f"[OK] {png_name}"
                )

            except Exception as e:

                print(
                    f"[ERROR] {fname}: {e}"
                )

    labels_csv = os.path.join(
        OUTPUT_DIR,
        "test_fixed_labels.csv"
    )

    pd.DataFrame(rows).to_csv(
        labels_csv,
        index=False
    )

    print(
        f"\nSaved labels to {labels_csv}"
    )


if __name__ == "__main__":
    main()
