import pandas as pd
import argparse
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score

parser = argparse.ArgumentParser(description='Evaluate pairwise metrics by criterion.')
parser.add_argument('--input_dir', '-i', type=str, required=True, help='Folder chứa các file .jsonl')
args = parser.parse_args()

files = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]

rows = []

for file in files:
    path = os.path.join(args.input_dir, file)

    df = pd.read_json(path, lines=True)

    # bỏ prediction lỗi
    df = df[df['prediction'].isin([1, 2])]

    if len(df) == 0:
        continue

    criteria_list = list(df['criteria'].unique())

    for criterion in criteria_list:

        sub_df = df[df['criteria'] == criterion]

        if len(sub_df) == 0:
            continue

        y_true = sub_df['label']
        y_pred = sub_df['prediction']

        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        df_A = sub_df[sub_df['label'] == 1]
        df_B = sub_df[sub_df['label'] == 2]

        acc_A = accuracy_score(df_A['label'], df_A['prediction']) if len(df_A) > 0 else 0
        acc_B = accuracy_score(df_B['label'], df_B['prediction']) if len(df_B) > 0 else 0

        rows.append({
            "model": file,
            "criterion": criterion,
            "acc": acc,
            "kappa": kappa,
            "acc_A": acc_A,
            "acc_B": acc_B
        })

# ===== to dataframe =====
df_res = pd.DataFrame(rows)

# ===== pivot =====
pivot_df = df_res.pivot(index="model", columns="criterion")

# swap level để: criterion → metric (đúng yêu cầu)
pivot_df = pivot_df.swaplevel(axis=1)

# sort cho đẹp
pivot_df = pivot_df.sort_index(axis=1, level=0)

print(pivot_df)

# optional: save
pivot_df.to_csv("pairwise_eval_pivot.csv")