import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend

# == directories ==

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_set")
PNG_DIR = os.path.join(OUTPUT_DIR, "pngs")

os.makedirs(PNG_DIR, exist_ok=True)


# == src data ===
WHALE_DIR = "../data/eq_data/fin_whale"
EARTHQUAKE_DIR = "../data/eq_data/das_data/earthquakes"
NOISE_DIR = "../data/eq_data/das_data/noise_data"


# == testing set , manually chosen ==

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

# === helpers ===
def find_first_dataset(group):
    """
    Recursively find the first dataset in an HDF5 file/group.
    """

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
        print(f"\n=== {os.path.basename(filepath)} ===")
        print("Keys:", list(f.keys()))

        if "data" in f:
            data = f["data"][:]
        else:
            data = find_first_dataset(f)
        if data is None:
            raise ValueError("No dataset found in file.")

        print("Shape:", data.shape)
    return data


def bandpass(data, lowcut, highcut, fs=100.0, order=4):
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def save_das_png(data, output_file):
    vmax = np.percentile(np.abs(data), 99)

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



def generate_png(filepath, label):
    data = load_h5_data(filepath)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    data = detrend(data, axis=1)

    if label == "whale":
        lowcut = 5.0
        highcut = 40.0

    else:
        lowcut = 2.0
        highcut = 10.0

    data = bandpass(
        data,
        lowcut=lowcut,
        highcut=highcut,
        fs=100.0
    )

    base = os.path.splitext(
        os.path.basename(filepath)
    )[0]

    png_file = os.path.join(
        PNG_DIR,
        f"{label}_{base}.png"
    )

    save_das_png(data, png_file)

    print(f"[PNG] {png_file}")


# ==== main =====
rows = []

for label, (source_dir, file_list) in DATASETS.items():
    for fname in file_list:
        filepath = os.path.join(
            source_dir,
            fname
        )

        if not os.path.exists(filepath):
            print(f"[MISSING] {filepath}")
            continue

        try:
            generate_png(
                filepath,
                label
            )

            rows.append(
                {
                    "file": fname,
                    "label": label.upper(),
                }
            )

        except Exception as e:
            print(
                f"[ERROR] {fname}: {e}"
            )


# == saving labels into csv
labels_csv = os.path.join(
    OUTPUT_DIR,
    "test_labels.csv"
)

pd.DataFrame(rows).to_csv(
    labels_csv,
    index=False
)

print("\n=========================")
print(f"Generated {len(rows)} PNGs")
print(f"Saved labels: {labels_csv}")
print("=========================")