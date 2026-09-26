"""
blocking.py — Modular blocking / candidate-generation methods for entity resolution.

Each blocking method is an independent function that takes preprocessed dataframes
and returns a dict: {source1_entity_id -> set of candidate entity_ids from S2/S3}.

All methods are recall-first — no precision optimization here.
"""
import pandas as pd
import numpy as np
import re
import os
import sys
import time
import logging
from collections import defaultdict
from typing import Dict, Set, Optional, List, Tuple

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# ===========================================================================
# Type alias for candidate sets
# ===========================================================================
CandidateMap = Dict[str, Set[str]]


# ===========================================================================
# (a) Token-overlap blocking on normalized business_name
# ===========================================================================
def token_overlap_blocking(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    name_col: str = 'norm_name',
    country_col: str = 'norm_country',
    min_token_overlap: int = 1,
    max_block_size: int = 5000,
    max_candidates_per_entity: int = 500,
) -> CandidateMap:
    """
    Inverted-index blocking: for each token in a business name, build an
    index of entity_ids containing that token (within the same country).
    For each S1 entity, retrieve all S2/S3 entities sharing at least
    `min_token_overlap` tokens in the same country.

    Always uses overlap counting to avoid memory blowup from unioning
    large blocks. Caps per-entity candidate set size.

    Args:
        s1_df: Preprocessed Source 1 dataframe
        s2s3_df: Preprocessed Source 2+3 dataframe (concatenated)
        name_col: Column with normalized name
        country_col: Column with normalized country
        min_token_overlap: Min number of shared tokens to be a candidate
        max_block_size: Skip tokens that appear in more than this many S2/S3 records
                        (these are stop-words that generate too many candidates)
        max_candidates_per_entity: Hard cap on candidates per S1 entity to prevent OOM
    
    Returns:
        CandidateMap: {s1_entity_id -> set of candidate s2/s3 entity_ids}
    """
    logger.info("Token-overlap blocking: building inverted index...")
    t0 = time.time()

    # Build inverted index: (token, country) -> list of s2s3 entity_ids
    inv_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for eid, name, country in zip(
        s2s3_df['entity_id'].values,
        s2s3_df[name_col].values,
        s2s3_df[country_col].values
    ):
        name_str = str(name) if pd.notna(name) else ''
        country_str = str(country) if pd.notna(country) else ''
        tokens = set(name_str.split())
        for tok in tokens:
            if len(tok) >= 2:  # skip single-char tokens
                inv_index[(tok, country_str)].append(eid)

    # Filter out oversized blocks (stop-word tokens)
    filtered_keys = []
    for key, eids in inv_index.items():
        if len(eids) > max_block_size:
            filtered_keys.append(key)
    for key in filtered_keys:
        del inv_index[key]

    logger.info(f"  Inverted index: {len(inv_index):,} (token, country) blocks "
                f"(filtered {len(filtered_keys)} oversized blocks). "
                f"Built in {time.time()-t0:.1f}s")

    # Query: for each S1 entity, count token overlap and select candidates
    candidates: CandidateMap = {}
    t1 = time.time()
    total = len(s1_df)

    for idx, (eid, name, country) in enumerate(zip(
        s1_df['entity_id'].values,
        s1_df[name_col].values,
        s1_df[country_col].values
    )):
        name_str = str(name) if pd.notna(name) else ''
        country_str = str(country) if pd.notna(country) else ''
        tokens = [tok for tok in set(name_str.split()) if len(tok) >= 2]

        # Always count overlap — never blindly union large blocks
        overlap_count: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            key = (tok, country_str)
            if key in inv_index:
                for cand_eid in inv_index[key]:
                    overlap_count[cand_eid] += 1

        # Filter by min overlap threshold
        if min_token_overlap <= 1:
            cands = set(overlap_count.keys())
        else:
            cands = {
                cand_eid for cand_eid, count in overlap_count.items()
                if count >= min_token_overlap
            }

        # Cap candidate set size — keep those with highest overlap
        if len(cands) > max_candidates_per_entity:
            sorted_cands = sorted(overlap_count.items(), key=lambda x: x[1], reverse=True)
            cands = {eid for eid, _ in sorted_cands[:max_candidates_per_entity]}

        candidates[eid] = cands

        if (idx + 1) % 5000 == 0:
            logger.info(f"  Token blocking: processed {idx+1:,}/{total:,} S1 entities "
                        f"({time.time()-t1:.1f}s)")

    logger.info(f"  Token blocking complete: {time.time()-t0:.1f}s total")
    return candidates


# ===========================================================================
# (b) Character n-gram TF-IDF cosine similarity with ANN (FAISS)
# ===========================================================================
def ngram_tfidf_blocking(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    name_col: str = 'norm_name_light',
    country_col: str = 'norm_country',
    ngram_range: Tuple[int, int] = (3, 4),
    top_k: int = 20,
    batch_size: int = 10000,
    max_features: int = 200000,
) -> CandidateMap:
    """
    Character n-gram TF-IDF + approximate nearest neighbor search.
    
    Uses lighter normalization (norm_name_light) to preserve character-level 
    signal that aggressive normalization destroys.
    
    Searches within same country to reduce the FAISS index size.
    
    Args:
        s1_df: Source 1 dataframe
        s2s3_df: Source 2+3 dataframe
        name_col: Column with lightly normalized name
        country_col: Column with normalized country
        ngram_range: Character n-gram range for TF-IDF
        top_k: Number of nearest neighbors to retrieve
        batch_size: Batch size for FAISS queries
        max_features: Max features for TF-IDF vectorizer
    
    Returns:
        CandidateMap
    """
    try:
        import faiss
    except ImportError:
        logger.warning("FAISS not installed — skipping n-gram TF-IDF blocking")
        return {}

    logger.info("N-gram TF-IDF blocking: fitting per-country vectorizers and FAISS indices...")
    t0 = time.time()

    candidates: CandidateMap = {}
    countries = sorted(set(s1_df[country_col].unique()) | set(s2s3_df[country_col].unique()))
    countries = [c for c in countries if c]  # remove empty

    for country in countries:
        logger.info(f"  Processing country: '{country}'")
        tc = time.time()

        s1_country = s1_df[s1_df[country_col] == country].copy()
        s2s3_country = s2s3_df[s2s3_df[country_col] == country].copy()

        if len(s1_country) == 0 or len(s2s3_country) == 0:
            logger.info(f"    Skipping — S1: {len(s1_country)}, S2S3: {len(s2s3_country)}")
            for eid in s1_country['entity_id'].values:
                candidates[eid] = set()
            continue

        # Build TF-IDF on combined corpus for consistent vocabulary
        all_names = pd.concat([
            s1_country[name_col].fillna(''),
            s2s3_country[name_col].fillna('')
        ]).values

        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=ngram_range,
            max_features=max_features,
            dtype=np.float32,
        )
        vectorizer.fit(all_names)

        # Transform S2/S3
        s2s3_vecs = vectorizer.transform(s2s3_country[name_col].fillna('').values)
        s2s3_eids = s2s3_country['entity_id'].values

        # Convert sparse to dense for FAISS (in float32)
        d = s2s3_vecs.shape[1]
        
        # Normalize for cosine similarity (use IP search = cosine after normalization)
        s2s3_dense = s2s3_vecs.toarray().astype(np.float32)
        norms = np.linalg.norm(s2s3_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        s2s3_dense /= norms

        # Build FAISS index
        index = faiss.IndexFlatIP(d)
        index.add(s2s3_dense)

        logger.info(f"    FAISS index built: {index.ntotal:,} vectors, dim={d}. "
                    f"({time.time()-tc:.1f}s)")

        # Free dense matrix memory
        del s2s3_dense

        # Query S1 in batches
        s1_names = s1_country[name_col].fillna('').values
        s1_eids = s1_country['entity_id'].values

        for batch_start in range(0, len(s1_eids), batch_size):
            batch_end = min(batch_start + batch_size, len(s1_eids))
            batch_names = s1_names[batch_start:batch_end]
            batch_eids = s1_eids[batch_start:batch_end]

            q_vecs = vectorizer.transform(batch_names).toarray().astype(np.float32)
            q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
            q_norms[q_norms == 0] = 1.0
            q_vecs /= q_norms

            actual_k = min(top_k, index.ntotal)
            _, indices = index.search(q_vecs, actual_k)

            for i, s1_eid in enumerate(batch_eids):
                cand_set = set()
                for j in indices[i]:
                    if j >= 0:
                        cand_set.add(s2s3_eids[j])
                candidates[s1_eid] = cand_set

        logger.info(f"    Country '{country}' done in {time.time()-tc:.1f}s")

        # Free memory
        del index, s2s3_vecs, vectorizer

    logger.info(f"  N-gram TF-IDF blocking complete: {time.time()-t0:.1f}s total")
    return candidates


# ===========================================================================
# (c) Locality / postal-code blocking on business_address
# ===========================================================================
def locality_blocking(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    address_col: str = 'norm_address',
    country_col: str = 'norm_country',
    max_block_size: int = 10000,
) -> CandidateMap:
    """
    Extract locality tokens from addresses and block on them.
    
    Strategy:
    1. Extract postal codes (5-6 digit sequences) if present
    2. Extract last 2-3 meaningful tokens from address as locality signal
       (city/state typically appear at the end of addresses across US/India/France)
    3. Block on (locality_token, country) pairs
    
    Falls back gracefully when address is missing — those entities simply
    don't generate candidates from this method (the union covers them).
    """
    logger.info("Locality blocking: extracting locality tokens...")
    t0 = time.time()

    # Regex for postal codes
    postal_re = re.compile(r'\b(\d{5,6})\b')

    def extract_locality_keys(address: str, country: str) -> Set[str]:
        """Extract blocking keys from an address."""
        keys = set()
        if not address or address == 'nan':
            return keys

        # 1. Postal code
        postal_matches = postal_re.findall(address)
        for pc in postal_matches:
            keys.add(f"postal:{pc}")

        # 2. Last 2-3 tokens as locality (city/state position)
        tokens = address.split()
        # Filter out very short tokens and pure numbers
        meaningful = [t for t in tokens if len(t) >= 3 and not t.isdigit()]
        if meaningful:
            # Take last 3 meaningful tokens
            for tok in meaningful[-3:]:
                keys.add(f"loc:{tok}")

        return keys

    # Build inverted index on S2/S3
    inv_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for eid, addr, country in zip(
        s2s3_df['entity_id'].values,
        s2s3_df[address_col].values,
        s2s3_df[country_col].values
    ):
        addr_str = str(addr) if pd.notna(addr) else ''
        country_str = str(country) if pd.notna(country) else ''
        loc_keys = extract_locality_keys(addr_str, country_str)
        for lk in loc_keys:
            inv_index[(lk, country_str)].append(eid)

    # Filter oversized blocks
    filtered = [k for k, v in inv_index.items() if len(v) > max_block_size]
    for k in filtered:
        del inv_index[k]
    logger.info(f"  Locality index: {len(inv_index):,} blocks "
                f"(filtered {len(filtered)} oversized). Built in {time.time()-t0:.1f}s")

    # Query S1
    candidates: CandidateMap = {}
    for eid, addr, country in zip(
        s1_df['entity_id'].values,
        s1_df[address_col].values,
        s1_df[country_col].values
    ):
        addr_str = str(addr) if pd.notna(addr) else ''
        country_str = str(country) if pd.notna(country) else ''
        loc_keys = extract_locality_keys(addr_str, country_str)
        cands = set()
        for lk in loc_keys:
            key = (lk, country_str)
            if key in inv_index:
                cands.update(inv_index[key])
        candidates[eid] = cands

    logger.info(f"  Locality blocking complete: {time.time()-t0:.1f}s total")
    return candidates


# ===========================================================================
# (d) Embedding-based semantic blocking (sentence-transformer + FAISS ANN)
# ===========================================================================
def embedding_blocking(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    name_col: str = 'norm_name_light',
    address_col: str = 'norm_address_light',
    country_col: str = 'norm_country',
    model_name: str = 'all-MiniLM-L6-v2',
    top_k: int = 20,
    batch_size: int = 512,
    faiss_nprobe: int = 32,
) -> CandidateMap:
    """
    Encode name+address with a small sentence embedding model, then ANN search.
    This is the primary generalization path for unseen countries (France).
    
    Uses an IVF FAISS index for scalability.
    
    Args:
        model_name: A small, open-license (Apache 2.0) sentence embedding model.
                    all-MiniLM-L6-v2 is ~22M params, MIT licensed.
        top_k: Number of nearest neighbors per S1 entity
        batch_size: Encoding batch size
        faiss_nprobe: Number of cells to probe in IVF index
    """
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers or FAISS not installed — skipping embedding blocking")
        return {}

    logger.info(f"Embedding blocking: loading model '{model_name}'...")
    t0 = time.time()

    model = SentenceTransformer(model_name)

    def make_text(row):
        """Combine name + address for embedding."""
        name = str(row[name_col]) if pd.notna(row[name_col]) else ''
        addr = str(row[address_col]) if pd.notna(row[address_col]) else ''
        parts = [p for p in [name, addr] if p]
        return ' '.join(parts) if parts else ''

    candidates: CandidateMap = {}

    # Process per-country (reduces index size, matches are always within country)
    countries = sorted(set(s1_df[country_col].unique()) | set(s2s3_df[country_col].unique()))
    countries = [c for c in countries if c]

    for country in countries:
        logger.info(f"  Embedding blocking for country: '{country}'")
        tc = time.time()

        s1_country = s1_df[s1_df[country_col] == country]
        s2s3_country = s2s3_df[s2s3_df[country_col] == country]

        if len(s1_country) == 0 or len(s2s3_country) == 0:
            for eid in s1_country['entity_id'].values:
                candidates[eid] = set()
            continue

        # Encode S2/S3
        s2s3_texts = s2s3_country.apply(make_text, axis=1).values.tolist()
        s2s3_eids = s2s3_country['entity_id'].values

        logger.info(f"    Encoding {len(s2s3_texts):,} S2/S3 texts...")
        s2s3_embeds = model.encode(
            s2s3_texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        # Build IVF FAISS index
        d = s2s3_embeds.shape[1]
        n_cells = min(int(np.sqrt(len(s2s3_embeds))), 4096)
        n_cells = max(n_cells, 1)

        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT)
        index.train(s2s3_embeds)
        index.add(s2s3_embeds)
        index.nprobe = faiss_nprobe

        logger.info(f"    FAISS IVF index built: {index.ntotal:,} vectors, "
                    f"{n_cells} cells. ({time.time()-tc:.1f}s)")

        del s2s3_embeds  # free memory

        # Encode and query S1 in batches
        s1_texts = s1_country.apply(make_text, axis=1).values.tolist()
        s1_eids = s1_country['entity_id'].values

        logger.info(f"    Encoding and querying {len(s1_texts):,} S1 texts...")
        for batch_start in range(0, len(s1_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(s1_texts))
            batch_texts = s1_texts[batch_start:batch_end]
            batch_eids = s1_eids[batch_start:batch_end]

            q_embeds = model.encode(
                batch_texts, batch_size=batch_size,
                normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)

            actual_k = min(top_k, index.ntotal)
            _, indices = index.search(q_embeds, actual_k)

            for i, s1_eid in enumerate(batch_eids):
                cand_set = set()
                for j in indices[i]:
                    if j >= 0:
                        cand_set.add(s2s3_eids[j])
                candidates[s1_eid] = cand_set

        logger.info(f"    Country '{country}' done in {time.time()-tc:.1f}s")
        del index

    logger.info(f"  Embedding blocking complete: {time.time()-t0:.1f}s total")
    return candidates


# ===========================================================================
# (e) Phonetic blocking (Soundex / Metaphone) — supplementary only
# ===========================================================================
def phonetic_blocking(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    name_col: str = 'norm_name',
    country_col: str = 'norm_country',
    max_block_size: int = 10000,
) -> CandidateMap:
    """
    Phonetic blocking using Metaphone on the first 2 meaningful name tokens.
    Supplementary method — never primary.
    
    Falls back gracefully if jellyfish is not installed.
    """
    try:
        import jellyfish
    except ImportError:
        logger.warning("jellyfish not installed — skipping phonetic blocking")
        return {}

    logger.info("Phonetic blocking: building metaphone index...")
    t0 = time.time()

    def get_phonetic_keys(name: str) -> List[str]:
        """Get metaphone codes for first 2 meaningful tokens."""
        tokens = [t for t in str(name).split() if len(t) >= 3 and t.isascii()]
        keys = []
        for tok in tokens[:2]:
            try:
                code = jellyfish.metaphone(tok)
                if code and len(code) >= 2:
                    keys.append(code)
            except Exception:
                pass
        return keys

    # Build inverted index
    inv_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for eid, name, country in zip(
        s2s3_df['entity_id'].values,
        s2s3_df[name_col].values,
        s2s3_df[country_col].values
    ):
        name_str = str(name) if pd.notna(name) else ''
        country_str = str(country) if pd.notna(country) else ''
        for key in get_phonetic_keys(name_str):
            inv_index[(key, country_str)].append(eid)

    # Filter oversized blocks
    filtered = [k for k, v in inv_index.items() if len(v) > max_block_size]
    for k in filtered:
        del inv_index[k]
    logger.info(f"  Phonetic index: {len(inv_index):,} blocks "
                f"(filtered {len(filtered)} oversized). Built in {time.time()-t0:.1f}s")

    # Query
    candidates: CandidateMap = {}
    for eid, name, country in zip(
        s1_df['entity_id'].values,
        s1_df[name_col].values,
        s1_df[country_col].values
    ):
        name_str = str(name) if pd.notna(name) else ''
        country_str = str(country) if pd.notna(country) else ''
        cands = set()
        for key in get_phonetic_keys(name_str):
            k = (key, country_str)
            if k in inv_index:
                cands.update(inv_index[k])
        candidates[eid] = cands

    logger.info(f"  Phonetic blocking complete: {time.time()-t0:.1f}s total")
    return candidates


# ===========================================================================
# Union combiner + candidate cap
# ===========================================================================
def combine_candidates(
    candidate_maps: Dict[str, CandidateMap],
    s1_entity_ids: List[str],
    max_candidates_per_entity: int = 200,
) -> CandidateMap:
    """
    Union all candidate maps and deduplicate.
    
    Args:
        candidate_maps: {method_name -> CandidateMap}
        s1_entity_ids: All S1 entity IDs (to ensure every S1 entity has a row)
        max_candidates_per_entity: Cap on candidate set size per S1 entity
    
    Returns:
        Combined CandidateMap with deduplication
    """
    logger.info(f"Combining {len(candidate_maps)} blocking methods via UNION...")

    combined: CandidateMap = {}
    for eid in s1_entity_ids:
        combined[eid] = set()

    for method_name, cmap in candidate_maps.items():
        for eid, cands in cmap.items():
            if eid in combined:
                combined[eid].update(cands)

    # Report stats before capping
    sizes = [len(v) for v in combined.values()]
    logger.info(f"  Before cap: avg={np.mean(sizes):.1f}, "
                f"median={np.median(sizes):.0f}, "
                f"max={np.max(sizes)}, "
                f"total_pairs={sum(sizes):,}")

    # Cap candidate set size
    if max_candidates_per_entity:
        capped = 0
        for eid in combined:
            if len(combined[eid]) > max_candidates_per_entity:
                # Keep a random subset — no precision optimization here
                combined[eid] = set(list(combined[eid])[:max_candidates_per_entity])
                capped += 1
        if capped > 0:
            logger.info(f"  Capped {capped:,} entities to max {max_candidates_per_entity} candidates")

    # Report stats after capping
    sizes = [len(v) for v in combined.values()]
    logger.info(f"  After cap: avg={np.mean(sizes):.1f}, "
                f"median={np.median(sizes):.0f}, "
                f"max={np.max(sizes)}, "
                f"total_pairs={sum(sizes):,}")

    return combined


# ===========================================================================
# Output writer
# ===========================================================================
def write_candidate_pairs(
    candidates: CandidateMap,
    output_path: str,
    s1_entity_ids: List[str],
):
    """
    Write candidate_pairs.tsv in the required format.
    
    One row per S1 entity, sorted by entity_id.
    candidate_entity_ids is comma-separated, empty string if no candidates.
    """
    logger.info(f"Writing candidate_pairs.tsv to {output_path}...")

    rows = []
    for eid in s1_entity_ids:
        cands = candidates.get(eid, set())
        cand_str = ','.join(sorted(cands)) if cands else ''
        rows.append({'source1_entity_id': eid, 'candidate_entity_ids': cand_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    logger.info(f"  Written {len(df):,} rows to {output_path}")
