import json
import os

class SubmissionDataset:
    def __init__(self, submissions_path="./data/submissions_test.jsonl", 
                 metadata_path="./data/metadata.jsonl",
                 limit=None): 
        self.submissions = []
        self.metadata_map = {}

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    meta = json.loads(line)
                    self.metadata_map[meta['id']] = meta

        # Load submissions
        if os.path.exists(submissions_path):
            with open(submissions_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if limit is not None and idx >= limit:
                        break
                    sub = json.loads(line)
                    sub_id = sub['sub_id']
                    id_parts = sub_id.rsplit('_', 1)
                    base_id = id_parts[0]
                    description = self.metadata_map.get(base_id, {}).get('description', '')
                    sub['description'] = description
                    self.submissions.append(sub)

    def __len__(self):
        return len(self.submissions)

    def __getitem__(self, idx):
        return self.submissions[idx]