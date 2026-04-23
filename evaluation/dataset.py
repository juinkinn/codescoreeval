import json
import os

class SubmissionDataset:
    def __init__(self, submissions_path="./data/submissions_test.jsonl", 
                 metadata_path="./data/metadata.jsonl",
                 limit=None): 
        self.submissions = []
        self.metadata_map = {}
        self.code_map = {}

        # Load code map
        code_path = "./data/submissions.jsonl"
        if os.path.exists(code_path):
            with open(code_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    self.code_map[item['sub_id']] = item

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

                    # ===== add code and lang =====
                    code_info = self.code_map.get(sub_id, {})
                    sub['code'] = code_info.get('code', '')
                    sub['lang'] = code_info.get('lang', '')
                    sub['correct'] = code_info.get('correct', None)

                    # ===== normalize gt scores =====
                    if 'correctness_score' in sub:
                        sub['gt_correctness_score'] = sub['correctness_score']
                    if 'efficiency_score' in sub:
                        sub['gt_efficiency_score'] = sub['efficiency_score']
                    if 'readability_score' in sub:
                        sub['gt_readability_score'] = sub['readability_score']
                    sub['output_format'] = meta.get('output_format', '')

                    self.submissions.append(sub)

    def __len__(self):
        return len(self.submissions)

    def __getitem__(self, idx):
        return self.submissions[idx]