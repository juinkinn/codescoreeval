import json
import argparse
from collections import defaultdict
from scipy.stats import rankdata

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

def get_pointwise_ranking(pointwise_data, criteria_list):
    # Method 1: Ranking base on invert pointwise score
    grouped = defaultdict(list)
    for row in pointwise_data:
        grouped[row["id"]].append(row)
        
    rankings = defaultdict(lambda: defaultdict(dict))
    scores_dict = defaultdict(lambda: defaultdict(dict))
    
    for problem_id, submissions in grouped.items():
        for criteria in criteria_list:
            score_key = f"{criteria}_score"
            sub_ids, scores = [], []
            
            for sub in submissions:
                if score_key in sub:
                    sid = sub["sub_id"]
                    sc = sub[score_key]
                    sub_ids.append(sid)
                    scores.append(sc)
                    scores_dict[problem_id][criteria][sid] = sc
            
            if scores:
                ranks = rankdata(scores, method='average')
                
                for sid, r in zip(sub_ids, ranks):
                    rankings[problem_id][criteria][sid] = r
                    
    return rankings, scores_dict

def get_pairwise_raw_scores(pairwise_data):
    # Method 2: Ranking based on counting wins/ties in pairwise
    raw_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for row in pairwise_data:
        pid = row["id"]
        crit = row["criteria"]
        s1 = row["sub_id_1"]
        s2 = row["sub_id_2"]
        label = row["label"]
        
        raw_scores[pid][crit][s1] += label
        raw_scores[pid][crit][s2] += (1 - label)
        
    return raw_scores

def main(args):
    criteria_list = ['correctness', 'efficiency', 'readability']
    results = []

    if args.mode == "pointwise":
        if not args.pointwise_path:
            raise ValueError("You need to provide --pointwise_path when using 'pointwise' mode")
        print(f"Loading Pointwise data from {args.pointwise_path}...")
        pw_data = load_jsonl(args.pointwise_path)
        pw_ranks, pw_scores = get_pointwise_ranking(pw_data, criteria_list)

    if args.mode == "pairwise":
        if not args.pairwise_path:
            raise ValueError("You need to provide --pairwise_path when using 'pairwise' mode")
        print(f"Loading Pairwise data from {args.pairwise_path}...")
        pair_data = load_jsonl(args.pairwise_path)
        pair_scores = get_pairwise_raw_scores(pair_data)

    if args.mode == "pointwise":
        for pid in pw_scores:
            for crit in criteria_list:
                for sid, score in pw_scores[pid][crit].items():
                    results.append({
                        "id": pid, "sub_id": sid, "criteria": crit,
                        "score": score,
                        "rank": pw_ranks[pid][crit][sid]
                    })

    elif args.mode == "pairwise":
        for pid in pair_scores:
            for crit in criteria_list:
                for sid, score in pair_scores[pid][crit].items():
                    results.append({
                        "id": pid, "sub_id": sid, "criteria": crit,
                        "rank": score
                    })

    save_jsonl(results, args.output_path)
    print(f"\nDone! Processed mode '{args.mode}'.")
    print(f"Saved {len(results)} records to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GT data based on selected mode")
    parser.add_argument("--mode", type=str, choices=["pointwise", "pairwise"], required=True,
                        help="Processing mode: pointwise, pairwise")
    parser.add_argument("--pointwise_path", type=str, default="", help="Path to file pointwise GT")
    parser.add_argument("--pairwise_path", type=str, default="", help="Path to file pairwise GT")
    parser.add_argument("--output_path", type=str, default="output_rankings.jsonl", help="File output JSONL")
    
    args = parser.parse_args()
    main(args)
