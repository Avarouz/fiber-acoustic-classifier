from dotenv import load_dotenv

import os
import base64
import pandas as pd

from openai import OpenAI


# ==== config ====
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REFERENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "reference_set",
    "pngs"
)

TEST_DIR = os.path.join(
    PROJECT_ROOT,
    "test_fixed",     # change this for different testing sets
    "pngs"
)

TEST_LABELS = os.path.join(
    PROJECT_ROOT,
    "test_fixed",     # also here !
    "test_fixed_labels.csv"       
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "predictions.csv"
)

client = OpenAI(api_key=API_KEY)



# === prompt === 

PROMPT = """
You are classifying DAS images.

There are three classes:

EARTHQUAKE
- Coherent signal across many channels
- Vertical or near-vertical arrivals
- Large spatial extent

WHALE
- Curved or hyperbolic moveout
- Localized in channel space
- Smooth coherent structure

NOISE
- No coherent moveout
- Random patterns
- Horizontal artifacts
- Instrument glitches

You will first see labeled reference examples.

Then you will see one unlabeled image.

Return ONLY one word:

EARTHQUAKE
WHALE
NOISE
"""


# ==== helper functions ====

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")


def load_reference_examples():
    refs = []

    for fname in sorted(os.listdir(REFERENCE_DIR)):
        if not fname.endswith(".png"):
            continue

        label = fname.split("_")[0].upper()

        refs.append(
            {
                "label": label,
                "path": os.path.join(
                    REFERENCE_DIR,
                    fname
                )
            }
        )

    return refs


def build_messages(test_image_path):
    '''
    creates the message to be fed into the LLM (prompt + 1 example of each signal type, with the label)
    '''
    content = [
        {
            "type": "text",
            "text": PROMPT,
        }
    ]

    references = load_reference_examples()

    for ref in references:         # add one per each example type
        content.append(
            {
                "type": "text",
                "text": f"REFERENCE EXAMPLE ({ref['label']})"
            }
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url":
                    f"data:image/png;base64,{encode_image(ref['path'])}"
                }
            }
        )

    content.append(
        {
            "type": "text",
            "text":
            "Classify the following image. Return only EARTHQUAKE, WHALE, or NOISE."
        }
    )

    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url":
                f"data:image/png;base64,{encode_image(test_image_path)}"
            }
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def classify_image(image_path):
    response = client.chat.completions.create(
        model=MODEL,
        messages=build_messages(image_path),
        temperature=0,
        max_tokens=5,
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )


# ======== MAIN ========
def main():
    gt_df = pd.read_csv(TEST_LABELS)

    results = []
    files = sorted(
        [
            f
            for f in os.listdir(TEST_DIR)
            if f.endswith(".png")
        ]
    )

    print(f"Testing {len(files)} images")

    for fname in files:
        image_path = os.path.join(
            TEST_DIR,
            fname
        )

        try:
            prediction = classify_image(
                image_path
            )

            truth = gt_df.loc[
                gt_df["file"] == fname,
                "label"
            ].values[0]

            print(
                f"{fname} -> {prediction} (GT={truth})"
            )

            results.append(
                {
                    "file": fname,
                    "prediction": prediction,
                    "ground_truth": truth,
                }
            )

        except Exception as e:

            print(
                f"ERROR: {fname}: {e}"
            )

    results_df = pd.DataFrame(results)

    accuracy = (
        results_df["prediction"]
        ==
        results_df["ground_truth"]
    ).mean()

    print("\n===================")
    print(f"Accuracy: {accuracy:.3f}")
    print("===================\n")

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"Saved predictions to {OUTPUT_CSV}"
    )

if __name__ == "__main__":
    main()