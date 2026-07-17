import json
import os


class SubmissionDataset:
    def __init__(
        self,
        submissions_path="../../data/submissions_test.jsonl",
        metadata_path="../../data/metadata.jsonl",
        code_path="../../data/submissions.jsonl",
        limit=None,
    ):
        self.submissions = []
        self.metadata_map = {}
        self.code_map = {}

        if os.path.exists(code_path):
            with open(code_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    self.code_map[item["sub_id"]] = item

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    meta = json.loads(line)
                    self.metadata_map[meta["id"]] = meta

        if os.path.exists(submissions_path):
            with open(submissions_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if limit is not None and idx >= limit:
                        break

                    sub = json.loads(line)
                    sub_id = sub["sub_id"]

                    base_id = sub_id.rsplit("_", 1)[0]

                    meta = self.metadata_map.get(base_id, {})
                    code_info = self.code_map.get(sub_id, {})

                    if not meta:
                        print(f"[WARN] Missing metadata for sub_id={sub_id}")

                    sub.update(
                        {
                            "description": meta.get("description", ""),
                            "title": meta.get("title", ""),
                            "constraints": meta.get("constraints", ""),
                            "input_format": meta.get("input_format", ""),
                            "output_format": meta.get("output_format", ""),
                            "code": code_info.get("code", ""),
                            "lang": code_info.get("lang", ""),
                            "correct": code_info.get("correct", None),
                        }
                    )

                    if "correctness_score" in sub:
                        sub["gt_correctness_score"] = sub["correctness_score"]
                    if "efficiency_score" in sub:
                        sub["gt_efficiency_score"] = sub["efficiency_score"]
                    if "readability_score" in sub:
                        sub["gt_readability_score"] = sub["readability_score"]

                    self.submissions.append(sub)

    def __len__(self):
        return len(self.submissions)

    def __getitem__(self, idx):
        return self.submissions[idx]