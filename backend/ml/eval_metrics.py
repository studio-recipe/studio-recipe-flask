import math
from typing import List, Set, Dict

def recall_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    topk = recommended[:k]
    hit = sum(1 for x in topk if x in ground_truth)
    return hit / float(len(ground_truth))

def dcg(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    score = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in ground_truth:
            score += 1.0 / math.log2(i + 1)
    return score

def ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    ideal = list(ground_truth)[:k]
    idcg = 0.0
    for i in range(1, min(k, len(ideal)) + 1):
        idcg += 1.0 / math.log2(i + 1)
    if idcg == 0:
        return 0.0
    return dcg(recommended, ground_truth, k) / idcg