import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = './data'
corr_chunk_1 = pd.read_json(f'{INPUT_DIR}/ground_truth/corr_chunk_1.jsonl', lines=True)
corr_chunk_2 = pd.read_json(f'{INPUT_DIR}/ground_truth/corr_chunk_2.jsonl', lines=True)
corr_chunk_3 = pd.read_json(f'{INPUT_DIR}/ground_truth/corr_chunk_3.jsonl', lines=True)
corr_chunk_4 = pd.read_json(f'{INPUT_DIR}/ground_truth/corr_chunk_4.jsonl', lines=True)

eff_chunk_1 = pd.read_json(f'{INPUT_DIR}/ground_truth/eff_chunk_1.jsonl', lines=True)
eff_chunk_2 = pd.read_json(f'{INPUT_DIR}/ground_truth/eff_chunk_2.jsonl', lines=True)
eff_chunk_3 = pd.read_json(f'{INPUT_DIR}/ground_truth/eff_chunk_3.jsonl', lines=True)
eff_chunk_4 = pd.read_json(f'{INPUT_DIR}/ground_truth/eff_chunk_4.jsonl', lines=True)

read_chunk_1 = pd.read_json(f'{INPUT_DIR}/ground_truth/read_chunk_1.jsonl', lines=True)
read_chunk_2 = pd.read_json(f'{INPUT_DIR}/ground_truth/read_chunk_2.jsonl', lines=True)
read_chunk_3 = pd.read_json(f'{INPUT_DIR}/ground_truth/read_chunk_3.jsonl', lines=True)
read_chunk_4 = pd.read_json(f'{INPUT_DIR}/ground_truth/read_chunk_4.jsonl', lines=True)

subset_corr = pd.read_json(f'{INPUT_DIR}/subset/correctness.json')

subset_eff_1 = pd.read_json(f'{INPUT_DIR}/subset/efficiency_100_1.json', lines=True)
subset_eff_2 = pd.read_json(f'{INPUT_DIR}/subset/efficiency_100_2.json', lines=True)

subset_read_1 = pd.read_json(f'{INPUT_DIR}/subset/syntax_100_1.json', lines=True)
subset_read_2 = pd.read_json(f'{INPUT_DIR}/subset/syntax_100_2.json', lines=True)

subset_eff = pd.concat([subset_eff_1, subset_eff_2], ignore_index=True)
subset_read = pd.concat([subset_read_1, subset_read_2], ignore_index=True)
subset_read.rename(columns={'syntax_score': 'readability_score'}, inplace=True)

corr = pd.concat([corr_chunk_1, corr_chunk_2, corr_chunk_3, corr_chunk_4, subset_corr], ignore_index=True)
eff = pd.concat([eff_chunk_1, eff_chunk_2, eff_chunk_3, eff_chunk_4, subset_eff], ignore_index=True)
read = pd.concat([read_chunk_1, read_chunk_2, read_chunk_3, read_chunk_4, subset_read], ignore_index=True)

final_df = (
    corr.merge(eff, on=['sub_id'], how='inner')
        .merge(read, on=['sub_id'], how='inner')
)

final_df.to_json(f'{INPUT_DIR}/final_dataset.jsonl', orient='records', lines=True)
print(f'Merged dataset saved to {INPUT_DIR}/final_dataset.jsonl')
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)
train_df.to_json(f'{INPUT_DIR}/train.jsonl', orient='records', lines=True)
print(f'Training set saved to {INPUT_DIR}/train.jsonl')
test_df.to_json(f'{INPUT_DIR}/test.jsonl', orient='records', lines=True)
print(f'Test set saved to {INPUT_DIR}/test.jsonl')
