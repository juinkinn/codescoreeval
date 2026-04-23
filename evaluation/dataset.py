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

                    # ===== robust match =====
                    base_id = None
                    if sub_id in self.metadata_map:
                        base_id = sub_id
                    else:
                        parts = sub_id.split('_')
                        for i in range(len(parts), 0, -1):
                            candidate = "_".join(parts[:i])
                            if candidate in self.metadata_map:
                                base_id = candidate
                                break

                    if base_id is None:
                        print(f"[WARN] Cannot match metadata for sub_id: {sub_id}")
                        meta = {}
                    else:
                        meta = self.metadata_map.get(base_id, {})

                    # ===== enrich submission =====
                    sub['description'] = meta.get('description', '')
                    sub['title'] = meta.get('title', '')
                    sub['constraints'] = meta.get('constraints', '')
                    sub['input_format'] = meta.get('input_format', '')
                    sub['output_format'] = meta.get('output_format', '')

                    self.submissions.append(sub)

    def __len__(self):
        return len(self.submissions)

    def __getitem__(self, idx):
        return self.submissions[idx]