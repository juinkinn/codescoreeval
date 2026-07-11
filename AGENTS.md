# Repository Notes

## Shape
- This is a script-and-data Python repo, not an installable package; most entrypoints are top-level scripts or files under `evaluation/`.
- There is no CI, test config, lint config, formatter config, or existing task runner in the repo.
- `README.md` only contains the project title; trust scripts and data paths over prose.

## Dependencies
- `requirements.txt` is UTF-16LE with CRLF line endings and only lists `datasets`, `ipykernel`, `pandas`, and `tqdm`.
- Scripts also import unpinned packages including `google-genai`, `python-dotenv`, `aiolimiter`, `torch`, `transformers`, `scikit-learn`, `scipy`, and `numpy`.

## Data Conventions
- Canonical data lives under `data/`; most scripts expect JSONL files and hardcode relative paths into this directory.
- Submission IDs derive the problem ID with `sub_id.rsplit("_", 1)[0]`; preserve that convention when creating or joining records.
- Criteria names are exactly `correctness`, `efficiency`, and `readability`.
- Pairwise labels/predictions use `1` for `sub_id_1` better, `0` for `sub_id_2` better, and `0.5` for tie.

## Common Commands
- Count Gemini prompt tokens: `python gemini-tokens-count.py -i data/submissions_test.jsonl -t all`; this writes `prompt_tokens.jsonl` in the current directory, not `data/prompt_tokens.jsonl`.
- Gemini scoring: `python gemini-inference.py -i data/submissions_test.jsonl --outptut output/pointwise/<name>.jsonl -t correctness`; the output flag is misspelled as `--outptut` in the script.
- Pairwise HF/local inference from repo root: `python evaluation/inference/infer_pairwise.py --model_name <hf-or-local-name> --limit 10`; add `--swapped` to create the swapped-output companion file.
- Pointwise HF/local inference has path-sensitive defaults; run it from `evaluation/inference/` or pass explicit data paths in code before using it from repo root.
- Build anchor test set from `evaluation/`: `python build_anchor_set.py`; its paths are `../data/...` relative to the current working directory.
- Evaluate pointwise outputs from repo root: `python evaluation/metric_pointwise_score.py -i output/pointwise/processed`.
- Evaluate pairwise classification from repo root: `python evaluation/metric_pairwise.py --pairwise_dir output/pairwise/processed --mode all`.
- Evaluate pairwise ranking from repo root: `python evaluation/metric_pairwise_ranking.py --pairwise_dir output/pairwise/processed --mode both`.
- Convert pointwise scores to pairwise predictions: `python evaluation/pointwise_to_pairwise.py -i <pointwise.jsonl> -o <pairwise.jsonl> -m llm`.

## Gotchas
- `gemini-inference.py` currently sets `api_key = ""` and comments out `os.getenv("GEMINI_API_KEY")`; fix that before expecting `.env` to work.
- `gemini-inference.py` filters to IDs from `data/failed_chunk_4.json`, but that file is not present in the current repo snapshot.
- `evaluation/inference/infer_pairwise.py` can reuse local model directories under `models/` for a few hardcoded HF names; otherwise it downloads from Hugging Face.
- Pairwise inference appends to existing output and skips completed `(id, criteria, sub_id_1, sub_id_2)` keys, so delete or rename old outputs when a fresh run is required.
- `fill_missing_choices.py` and `fill_missing_scores.py` fill missing values randomly without a fixed seed; do not use them when deterministic outputs matter unless you add seeding.
