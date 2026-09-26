"""
evaluate_blocking.py — Recall@blocking evaluation harness.

Computes per-method and combined recall, plus reduction ratio stats,
on a held-out validation split from train_ground_truth.tsv.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Set, List, Tuple

logger = logging.getLogger(__name__)

CandidateMap = Dict[str, Set[str]]


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Parse ground truth into {s1_entity_id -> set of true matched entity_ids}.
    Singletons (no matches) map to empty set.
    """
    truth = {}
    for _, row in gt_df.iterrows():
        s1_eid = row['source1_entity_id']
        matched = str(row.get('matched_entity_ids', ''))
        if matched and matched != 'nan' and matched.strip():
            truth[s1_eid] = set(matched.split(','))
        else:
            truth[s1_eid] = set()
    return truth


def compute_blocking_recall(
    candidates: CandidateMap,
    truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Compute recall@blocking: fraction of true matches that survive
    into the candidate set.
    
    Returns dict with:
        - recall: overall recall
        - recall_non_singleton: recall over entities that have at least 1 true match
        - avg_candidate_size: average candidate set size
        - median_candidate_size: median candidate set size
        - max_candidate_size: max candidate set size
        - total_pairs: total number of candidate pairs
        - coverage: fraction of S1 entities with non-empty candidate set
        - n_entities: number of S1 entities evaluated
        - n_entities_with_matches: number of S1 entities with true matches
    """
    total_true_matches = 0
    total_recovered = 0
    total_true_matches_nonsingle = 0
    total_recovered_nonsingle = 0
    candidate_sizes = []
    n_with_matches = 0

    for s1_eid, true_matches in truth.items():
        cands = candidates.get(s1_eid, set())
        candidate_sizes.append(len(cands))

        if true_matches:
            n_with_matches += 1
            recovered = len(true_matches & cands)
            total_true_matches += len(true_matches)
            total_recovered += recovered
            total_true_matches_nonsingle += len(true_matches)
            total_recovered_nonsingle += recovered

    recall = total_recovered / total_true_matches if total_true_matches > 0 else 0.0
    recall_ns = (total_recovered_nonsingle / total_true_matches_nonsingle
                 if total_true_matches_nonsingle > 0 else 0.0)

    candidate_sizes_arr = np.array(candidate_sizes) if candidate_sizes else np.array([0])

    return {
        'recall': recall,
        'recall_non_singleton': recall_ns,
        'avg_candidate_size': float(np.mean(candidate_sizes_arr)),
        'median_candidate_size': float(np.median(candidate_sizes_arr)),
        'max_candidate_size': int(np.max(candidate_sizes_arr)),
        'total_pairs': int(np.sum(candidate_sizes_arr)),
        'coverage': float(np.mean(candidate_sizes_arr > 0)),
        'n_entities': len(truth),
        'n_entities_with_matches': n_with_matches,
    }


def evaluate_all_methods(
    method_candidates: Dict[str, CandidateMap],
    combined_candidates: CandidateMap,
    truth: Dict[str, Set[str]],
) -> pd.DataFrame:
    """
    Evaluate all individual blocking methods + combined, print a comparison table.
    
    Args:
        method_candidates: {method_name -> CandidateMap}
        combined_candidates: union CandidateMap
        truth: parsed ground truth
    
    Returns:
        DataFrame with recall/stats per method
    """
    rows = []

    for method_name, cands in method_candidates.items():
        stats = compute_blocking_recall(cands, truth)
        stats['method'] = method_name
        rows.append(stats)

    # Combined
    stats = compute_blocking_recall(combined_candidates, truth)
    stats['method'] = '*** COMBINED (union) ***'
    rows.append(stats)

    df = pd.DataFrame(rows)
    cols = ['method', 'recall', 'recall_non_singleton', 'avg_candidate_size',
            'median_candidate_size', 'max_candidate_size', 'total_pairs', 'coverage']
    df = df[cols]

    print("\n" + "=" * 120)
    print("BLOCKING EVALUATION RESULTS")
    print("=" * 120)
    print(df.to_string(index=False, float_format='%.4f'))
    print("=" * 120 + "\n")

    return df


def train_val_split(
    gt_df: pd.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ground truth into train and validation.
    Returns (train_gt, val_gt) DataFrames.
    """
    np.random.seed(seed)
    n = len(gt_df)
    val_size = int(n * val_fraction)
    indices = np.random.permutation(n)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return gt_df.iloc[train_indices].copy(), gt_df.iloc[val_indices].copy()
