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


# ==== reference set =====
REFERENCE_DATASETS = {
    "whale": (
        WHALE_DIR,
        [
            "signal_66582.h5",
            "signal_1131.h5",   # weak
            "signal_2125.h5",   # REALLY weak
        ],
    ),
    "earthquake": (
        EARTHQUAKE_DIR,
        [
            "ci39007775.h5",
            "ci38971232.h5",
            "nn00832186-1.h5",  # weird

        ],
    ),
    "noise": (
        NOISE_DIR,
        [
            "ci39812319-2.h5",
            "ci39281440.h5",    # coherent
            "ci38538991.h5",

        ],
    ),
}


# == testing set , manually chosen ==

TEST_DATASETS = {
    "whale": (
        WHALE_DIR,
        [
            "signal_7402.h5",
            "signal_9441.h5",
            "signal_11183.h5",
            "signal_20099.h5",
            "signal_20151.h5",

            "signal_9660.h5",
            "signal_2138.h5",
            "signal_9704.h5",
            "signal_9414.h5",
            "signal_20175.h5",

            "signal_9660.h5",
            "signal_2097.h5"
            "signal_38865.h5",
            "signal_16446.h5",
            "signal_17195.h5",

            "signal_5527.h5",
            "signal_1822.h5",
            "signal_1834.h5",
            "signal_20131.h5",
            "signal_28287.h5",
        ],
    ),

    "earthquake": (
        EARTHQUAKE_DIR,
        [
            "ci38628799.h5",
            "ci38538991.h5",
            "ci39227895.h5",
            "ci38242914.h5",
            "nn00787309.h5",

            "nc73585086.h5",
            "ci38593535.h5",
            "nn00804187-1.h5",
            "ci38557895.h5",
            "nn00787479.h5",

            "ci38387018.h5",
            "ci39217768.h5",
            "ci39759376-1.h5",
            "ci39469848.h5",
            "ci39014215.h5",

            "ci39835791.h5",
            "ci38996632.h5",
            "ci39322287.h5",
            "ci40151807.h5",
            "ci39550567-1.h5",

        ],
    ),

    "noise": (
        NOISE_DIR,
        [
            "ci38529591.h5",
            "nn00831033.h5",
            "NCSN73972711.0.h5",
            "NCSN73774300.0.h5",
            "ci40024128.h5",

            "NCSN73902356.0.h5",
            "us7000i9d9.h5",
            "nc73615986-2.h5",
            "us7000clvy.h5",
            "NCSN73765585.0.h5",

            "ci40027143.h5",
            "nn00806025.h5",
            "NCSN73982331.0.h5",
            "nc73585291-1.h5",
            "nc73566711-1.h5",

            "nn00788443-1.h5",
            "ci40069263.h5",
            "ci39928664-1.h5",
            "ci38644943.h5",
            "ci39406880.h5",
        ],
    ),
}


# ====== helpers ======
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
# 

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


def generate_png(filepath, label, png_dir):
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
        fs=100.0,
    )

    base = os.path.splitext(
        os.path.basename(filepath)
    )[0]

    png_file = os.path.join(
        png_dir,
        f"{label}_{base}.png"
    )

    save_das_png(
        data,
        png_file
    )

    print(f"[PNG] {png_file}")



def build_dataset(dataset_name, datasets):

    output_dir = os.path.join(
        PROJECT_ROOT,
        dataset_name
    )

    png_dir = os.path.join(
        output_dir,
        "pngs"
    )

    os.makedirs(
        png_dir,
        exist_ok=True
    )

    rows = []

    for label, (source_dir, file_list) in datasets.items():
        for fname in file_list:
            filepath = os.path.join(
                source_dir,
                fname
            )

            if not os.path.exists(filepath):
                print(
                    f"[MISSING] {filepath}"
                )

                continue

            try:
                generate_png(
                    filepath,
                    label,
                    png_dir
                )

                rows.append(
                    {
                        "file": f"{label}_{os.path.splitext(fname)[0]}.png",
                        "label": label.upper(),
                    }
                )

            except Exception as e:

                print(
                    f"[ERROR] {fname}: {e}"
                )

    # === save labels into csv ====
    labels_csv = os.path.join(
        output_dir,
        f"{dataset_name}_labels.csv"
    )

    pd.DataFrame(rows).to_csv(
        labels_csv,
        index=False
    )

    print(
        f"\nSaved labels: {labels_csv}"
    )



# ======= MAIN =======
if __name__ == "__main__":

    build_dataset(
        "reference_set",
        REFERENCE_DATASETS
    )

    build_dataset(
        "test_set",
        TEST_DATASETS
    )
