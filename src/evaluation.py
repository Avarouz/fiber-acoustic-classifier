import pandas as pd

CSV_PATH = "labels.csv"
df = pd.read_csv(CSV_PATH)

def true_label(fname):
    fname = fname.lower()
    if "noise" in fname:
        return "NOISE"
    elif "whale" in fname:
        return "WHALE"
    else:
        return "EARTHQUAKE"


df["true"] = df["file"].apply(true_label)
df["pred"] = df["label"].str.upper().str.strip()
df["correct"] = df["true"] == df["pred"]

# ===== accuracy =====
accuracy = df["correct"].mean()
print(f"Accuracy: {accuracy:.3f} ({df['correct'].sum()}/{len(df)})")


# ===== per-class accuracy =====
print("\nPer-class accuracy:")
for cls in ["EARTHQUAKE", "WHALE", "NOISE"]:
    subset = df[df["true"] == cls]
    if len(subset) == 0:
        continue
    acc = subset["correct"].mean()
    print(f"  {cls}: {acc:.3f} ({subset['correct'].sum()}/{len(subset)})")



# ===== Confusion matrix (text) =====
print("\nConfusion matrix (true → predicted):")
print(df.groupby(["true", "pred"]).size().to_string())



# ===== Errors =====
errors = df[~df["correct"]][["file", "true", "pred"]]
print(f"\nErrors ({len(errors)}):")
print(errors.to_string(index=False))

df.to_csv("labels_with_eval.csv", index=False)
print("\nSaved: labels_with_eval.csv")