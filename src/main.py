from dotenv import load_dotenv

import os
import base64
import pandas as pd

from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
IMAGE_DIR = "../image-data"
OUTPUT_CSV = "labels.csv"
MODEL = "gpt-4o"

client = OpenAI(api_key=API_KEY)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    

PROMPT = """You are an expert analyst classifying Distributed Acoustic Sensing (DAS) array plots.
The plot shows normalized strain amplitude across fiber channels (y-axis) over time (x-axis).

Return ONLY one word: EARTHQUAKE, WHALE, or NOISE

--- SIGNAL SIGNATURES ---

EARTHQUAKE:
- Sudden onset: energy appears simultaneously or near-simultaneously across MANY channels at once
- Vertical or near-vertical wavefront at onset (fast propagation)
- Followed by coda: sustained, decaying energy after the initial arrival
- May show multiple wave phases (P, S) arriving at slightly different times
- Energy is broadband and affects large channel ranges (hundreds of channels)
- Example: a dark vertical band that fans out and decays

WHALE:
- Localized in channel space: affects a limited range of channels, not the whole array
- Hyperbolic or curved moveout: signal arrives at different times on different channels,
  forming a curved or diagonal streak (not vertical)
- May show a fan shape with an apex (closest point of approach) and trailing tail
- Energy is relatively narrow-band and smooth, not impulsive
- Signal drifts slowly across channels over many seconds
- Example: a curved smear or diagonal slash confined to a channel sub-range

NOISE:
- No coherent onset across multiple channels
- Purely horizontal streaks: single bad channels with constant noise (cable artifact)
- Random speckle with no spatial or temporal coherence
- Signals confined to 1-2 channels only
- Rectangular blocks of uniform amplitude: data gaps or instrument glitches

--- DECISION RULES ---
1. Is there a sudden vertical onset across hundreds of channels? → EARTHQUAKE
2. Is there a curved/hyperbolic moveout confined to a channel subrange? → WHALE
3. Is the pattern diagonal streaks, horizontal lines, random speckle, or data blocks? → NOISE
4. If ambiguous between WHALE and NOISE: does it have any curved moveout structure? Yes → WHALE
5. If ambiguous between EARTHQUAKE and NOISE: does it affect many channels at once? Yes → EARTHQUAKE

Return only: EARTHQUAKE, WHALE, or NOISE"""

def label_image(image_path):
    image_base64 = encode_image(image_path)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip().upper()

def main():
    results = []
    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
    print(f"Found {len(files)} images")
    for fname in files:
        fpath = os.path.join(IMAGE_DIR, fname)
        try:
            label = label_image(fpath)
            print(f"{fname} → {label}")
            results.append({"file": fname, "label": label})
        except Exception as e:
            print(f"Error on {fname}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved labels to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()