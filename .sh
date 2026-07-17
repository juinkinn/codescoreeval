# I. Test set

# Pointwise scoring
!python .\evaluation\metric_pointwise_score.py -i .\output\pointwise\processed
# Pairwise scoring
!python .\evaluation\metric_pairwise_score.py --pairwise_dir .\output\pairwise\processed\

# Convert pointwise to pairwise ranking
!python .\evaluation\pointwise_to_pairwise.py -i .\output\pointwise\processed -o .\output\pairwise\processed -m both
# Pairwise ranking
!python .\evaluation\metric_pairwise_ranking.py --pairwise_dir .\output\pairwise\processed\ --mode all

# II. Human set

# Pointwise scoring
!python .\evaluation\metric_pointwise_score.py -i .\output\human_evaluation\pointwise\ -a .\data\human_annotated_test_set\pairwise_human_test_with_anchors.jsonl
# Pairwise scoring
!python .\evaluation\metric_pairwise_score.py --pairwise_dir .\output\human_evaluation\pairwise\ --anchor_path .\data\human_annotated_test_set\pairwise_human_test_with_anchors.jsonl --test_path .\data\human_annotated_test_set\pointwise_human_test_set.jsonl
# Convert pointwise to pairwise ranking
!python .\evaluation\pointwise_to_pairwise.py -i .\output\human_evaluation\pointwise\ -o .\output\human_evaluation\pairwise\ -m both
# Pairwise ranking
!python .\evaluation\metric_pairwise_ranking.py --pairwise_dir .\output\human_evaluation\pairwise\ --mode all
