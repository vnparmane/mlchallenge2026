"""
run_blocking.py — Main orchestrator for the blocking pipeline.

Usage:
    # Evaluate on validation split (default: 10% of training data)
    python src/run_blocking.py --mode eval --val-fraction 0.1

    # Generate candidate_pairs.tsv for the test set
    python src/run_blocking.py --mode test

    # Both
    python src/run_blocking.py --mode both
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

# Ensure the src directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_source, load_ground_truth
from blocking import (
    token_overlap_blocking,
    ngram_tfidf_blocking,
    locality_blocking,
    embedding_blocking,
    phonetic_blocking,
    combine_candidates,
    write_candidate_pairs,
)
from evaluate_blocking import (
    parse_ground_truth,
    evaluate_all_methods,
    train_val_split,
)

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


def get_paths(base_dir: str):
    """Return a dict of all relevant file paths."""
    return {
        'train_s1': os.path.join(base_dir, 'dataset', 'train', 'train_source1.tsv'),
        'train_s2': os.path.join(base_dir, 'dataset', 'train', 'train_source2.tsv'),
        'train_s3': os.path.join(base_dir, 'dataset', 'train', 'train_source3.tsv'),
        'train_gt': os.path.join(base_dir, 'dataset', 'train', 'train_ground_truth.tsv'),
        'test_s1': os.path.join(base_dir, 'dataset', 'test', 'test_source1.tsv'),
        'test_s2': os.path.join(base_dir, 'dataset', 'test', 'test_source2.tsv'),
        'test_s3': os.path.join(base_dir, 'dataset', 'test', 'test_source3.tsv'),
        'output_dir': os.path.join(base_dir, 'output'),
    }


def run_all_blocking_methods(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    enable_embedding: bool = True,
    enable_ngram_tfidf: bool = True,
    enable_phonetic: bool = True,
):
    """
    Run all enabled blocking methods and return individual + combined results.
    
    Returns:
        (method_candidates: dict, combined_candidates: CandidateMap)
    """
    method_candidates = {}

    # (a) Token-overlap blocking — always enabled, fast and effective
    logger.info("\n>>> Running Token-Overlap Blocking <<<")
    method_candidates['token_overlap'] = token_overlap_blocking(
        s1_df, s2s3_df,
        name_col='norm_name',
        country_col='norm_country',
        min_token_overlap=1,
    )

    # (c) Locality blocking — always enabled, fast
    logger.info("\n>>> Running Locality Blocking <<<")
    method_candidates['locality'] = locality_blocking(
        s1_df, s2s3_df,
        address_col='norm_address',
        country_col='norm_country',
    )

    # (e) Phonetic blocking — supplementary
    if enable_phonetic:
        logger.info("\n>>> Running Phonetic Blocking <<<")
        result = phonetic_blocking(
            s1_df, s2s3_df,
            name_col='norm_name',
            country_col='norm_country',
        )
        if result:
            method_candidates['phonetic'] = result

    # (b) N-gram TF-IDF blocking — computationally heavier
    if enable_ngram_tfidf:
        logger.info("\n>>> Running N-gram TF-IDF Blocking <<<")
        result = ngram_tfidf_blocking(
            s1_df, s2s3_df,
            name_col='norm_name_light',
            country_col='norm_country',
            ngram_range=(3, 4),
            top_k=20,
            max_features=200000,
        )
        if result:
            method_candidates['ngram_tfidf'] = result

    # (d) Embedding blocking — most expensive, best for generalization
    if enable_embedding:
        logger.info("\n>>> Running Embedding Blocking <<<")
        result = embedding_blocking(
            s1_df, s2s3_df,
            name_col='norm_name_light',
            address_col='norm_address_light',
            country_col='norm_country',
            model_name='all-MiniLM-L6-v2',
            top_k=20,
        )
        if result:
            method_candidates['embedding'] = result

    # Combine via union
    logger.info("\n>>> Combining all methods via UNION <<<")
    s1_eids = s1_df['entity_id'].tolist()
    combined = combine_candidates(
        method_candidates,
        s1_entity_ids=s1_eids,
        max_candidates_per_entity=200,
    )

    return method_candidates, combined


def run_eval(paths: dict, val_fraction: float = 0.1,
             enable_embedding: bool = False, enable_ngram_tfidf: bool = True):
    """Run blocking on a validation split and evaluate recall."""
    logger.info("=" * 80)
    logger.info("EVALUATION MODE: running on validation split")
    logger.info("=" * 80)

    t0 = time.time()

    # Load ground truth and split
    logger.info("Loading ground truth...")
    gt = load_ground_truth(paths['train_gt'])
    _, val_gt = train_val_split(gt, val_fraction=val_fraction)
    val_s1_eids = set(val_gt['source1_entity_id'].values)
    logger.info(f"Validation split: {len(val_gt):,} S1 entities "
                f"({val_fraction*100:.0f}% of training data)")

    # Parse truth for the validation set
    truth = parse_ground_truth(val_gt)

    # Get all true match IDs from val set (to figure out which S2/S3 we need)
    val_s2s3_match_ids = set()
    for matches in truth.values():
        val_s2s3_match_ids.update(matches)

    # Load sources
    logger.info("Loading and preprocessing Source 1...")
    s1_df = load_source(paths['train_s1'])
    # Filter to validation S1 entities only
    s1_df = s1_df[s1_df['entity_id'].isin(val_s1_eids)].reset_index(drop=True)
    logger.info(f"  S1 validation: {len(s1_df):,} entities")

    logger.info("Loading and preprocessing Source 2...")
    s2_df = load_source(paths['train_s2'])
    logger.info(f"  S2: {len(s2_df):,} entities")

    logger.info("Loading and preprocessing Source 3...")
    s3_df = load_source(paths['train_s3'])
    logger.info(f"  S3: {len(s3_df):,} entities")

    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)
    logger.info(f"  S2+S3 combined: {len(s2s3_df):,} entities")

    del s2_df, s3_df  # free memory

    # Run blocking
    method_candidates, combined = run_all_blocking_methods(
        s1_df, s2s3_df,
        enable_embedding=enable_embedding,
        enable_ngram_tfidf=enable_ngram_tfidf,
    )

    # Evaluate
    eval_df = evaluate_all_methods(method_candidates, combined, truth)

    elapsed = time.time() - t0
    logger.info(f"Evaluation completed in {elapsed/60:.1f} minutes")

    return eval_df


def run_test(paths: dict, enable_embedding: bool = False,
             enable_ngram_tfidf: bool = True):
    """Run blocking on the test set and write candidate_pairs.tsv."""
    logger.info("=" * 80)
    logger.info("TEST MODE: generating candidate_pairs.tsv for test set")
    logger.info("=" * 80)

    t0 = time.time()

    # Load test sources
    logger.info("Loading and preprocessing test Source 1...")
    s1_df = load_source(paths['test_s1'])
    logger.info(f"  S1: {len(s1_df):,} entities")

    logger.info("Loading and preprocessing test Source 2...")
    s2_df = load_source(paths['test_s2'])
    logger.info(f"  S2: {len(s2_df):,} entities")

    logger.info("Loading and preprocessing test Source 3...")
    s3_df = load_source(paths['test_s3'])
    logger.info(f"  S3: {len(s3_df):,} entities")

    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)
    logger.info(f"  S2+S3 combined: {len(s2s3_df):,} entities")

    del s2_df, s3_df

    # Run blocking
    method_candidates, combined = run_all_blocking_methods(
        s1_df, s2s3_df,
        enable_embedding=enable_embedding,
        enable_ngram_tfidf=enable_ngram_tfidf,
    )

    # Write output
    os.makedirs(paths['output_dir'], exist_ok=True)
    output_path = os.path.join(paths['output_dir'], 'candidate_pairs.tsv')
    s1_eids = s1_df['entity_id'].tolist()
    write_candidate_pairs(combined, output_path, s1_eids)

    elapsed = time.time() - t0
    logger.info(f"Test candidate generation completed in {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Entity Resolution Blocking Pipeline')
    parser.add_argument('--mode', choices=['eval', 'test', 'both'], default='eval',
                        help='Mode: eval (validation), test (generate output), both')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Fraction of training data to use for validation (default: 0.1)')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory (student_resource/). Auto-detected if not specified.')
    parser.add_argument('--enable-embedding', action='store_true', default=False,
                        help='Enable embedding-based blocking (slow, requires GPU ideally)')
    parser.add_argument('--disable-ngram-tfidf', action='store_true', default=False,
                        help='Disable n-gram TF-IDF blocking')
    args = parser.parse_args()

    # Auto-detect base dir
    if args.base_dir is None:
        # Assume we're running from code/business_entity_resolution/
        src_dir = os.path.dirname(os.path.abspath(__file__))
        args.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(src_dir)))

    paths = get_paths(args.base_dir)

    # Verify paths exist
    for key in ['train_s1', 'train_s2', 'train_s3', 'train_gt']:
        if not os.path.exists(paths[key]):
            logger.error(f"Missing file: {paths[key]}")
            sys.exit(1)

    enable_ngram = not args.disable_ngram_tfidf

    if args.mode in ('eval', 'both'):
        run_eval(paths, val_fraction=args.val_fraction,
                 enable_embedding=args.enable_embedding,
                 enable_ngram_tfidf=enable_ngram)

    if args.mode in ('test', 'both'):
        for key in ['test_s1', 'test_s2', 'test_s3']:
            if not os.path.exists(paths[key]):
                logger.error(f"Missing test file: {paths[key]}")
                sys.exit(1)
        run_test(paths, enable_embedding=args.enable_embedding,
                 enable_ngram_tfidf=enable_ngram)


if __name__ == '__main__':
    main()
