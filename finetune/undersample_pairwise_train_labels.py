import pandas as pd

INPUT_FILE = "pairwise_train_labels.jsonl"
OUTPUT_FILE = "pairwise_train_labels_balanced.jsonl"

RANDOM_STATE = 42
TARGET = 30000 

df = pd.read_json(INPUT_FILE, lines=True)

print("Original:")
print(df["label"].value_counts().sort_index())
criteria_count = df["criteria"].nunique()
TARGET_PER_CRITERIA = TARGET // criteria_count

label0 = df[df["label"] == 0.0].copy()
label1 = df[df["label"] == 1.0].copy()
label05 = df[df["label"] == 0.5].copy()

all_pair = pd.concat([label0, label1], ignore_index=True)

flip = all_pair.copy()

flip[["sub_id_1", "sub_id_2"]] = flip[["sub_id_2", "sub_id_1"]].values
flip["label"] = flip["label"].map({0.0: 1.0, 1.0: 0.0})

pool = pd.concat([all_pair, flip], ignore_index=True)

pool0 = pool[pool["label"] == 0.0]
pool1 = pool[pool["label"] == 1.0]

pool0 = pool0.groupby("criteria").sample(n=TARGET_PER_CRITERIA, random_state=RANDOM_STATE)
pool1 = pool1.groupby("criteria").sample(n=TARGET_PER_CRITERIA, random_state=RANDOM_STATE)

final_df = pd.concat([pool0, pool1, label05], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nFINAL:")
print(final_df["label"].value_counts().sort_index())

print("\nBy criteria:")
print(
    final_df.groupby(["criteria", "label"])
    .size()
    .unstack(fill_value=0)
)

final_df.to_json(
    OUTPUT_FILE,
    orient="records",
    lines=True,
    force_ascii=False
)

print("\nSaved:", OUTPUT_FILE)