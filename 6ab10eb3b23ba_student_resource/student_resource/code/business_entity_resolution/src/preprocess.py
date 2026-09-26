"""
preprocess.py — Data-driven preprocessing for entity resolution.

Normalizes business names and addresses using patterns mined from training data
rather than hardcoded external gazetteers. Country-agnostic throughout.
"""
import pandas as pd
import re
import unicodedata
from collections import Counter


# ---------------------------------------------------------------------------
# Legal suffix patterns — mined from token-frequency analysis of training data.
# These are the top-frequency tokens that are legal-entity suffixes, not
# meaningful business name content. We strip them to reduce noise during
# blocking. Ordered roughly by frequency in train_source1.
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    "limited", "private", "llc", "inc", "ltd", "pvt",
    "corp", "llp", "pllc", "lp", "corporation", "company",
    "co", "pc", "incorporated", "plc", "sa", "sarl",
    "sas", "gmbh", "ag", "srl", "pty",
    # Punctuated variants get normalized to these after punctuation stripping
}

# Patterns that appear as "(india)", "(us)", etc. — country tags in names
PAREN_TAG_RE = re.compile(r'\([^)]*\)')

# ---------------------------------------------------------------------------
# Address abbreviations — common pairs observed across US/India training data.
# Applied uniformly regardless of country.
# ---------------------------------------------------------------------------
ADDRESS_ABBREVS = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bsq\b': 'square',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\brm\b': 'room',
    r'\bbldg\b': 'building',
    r'\bfl\b': 'floor',
    r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway',
    r'\bcir\b': 'circle',
    r'\bexpy\b': 'expressway',
    r'\bext\b': 'extension',
    r'\bno\b': 'number',
    r'\bdist\b': 'district',
    r'\btq\b': 'taluk',
    r'\bblk\b': 'block',
    r'\bkh\b': 'khasra',
}

# Compiled regex for address abbreviations
_ADDR_ABBREV_COMPILED = [(re.compile(pat), repl) for pat, repl in ADDRESS_ABBREVS.items()]


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode: NFC normalization preserves accented characters 
    (important for French) while standardizing representation.
    """
    if not text:
        return ""
    # NFC normalization — composes characters, keeps accents intact
    return unicodedata.normalize('NFC', text)


def clean_text_light(text: str) -> str:
    """
    Light cleaning: lowercase, normalize unicode, collapse whitespace.
    Preserves accented characters and most punctuation.
    Used for character n-gram blocking where we want character-level signal.
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    if not text or text.lower() == 'nan':
        return ""
    text = normalize_unicode(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text(text: str) -> str:
    """
    Aggressive cleaning: lowercase, strip all non-alphanumeric (keeping 
    unicode letters for Hindi/French), collapse whitespace.
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    if not text or text.lower() == 'nan':
        return ""
    text = normalize_unicode(text).lower()
    # Remove parenthetical tags like (india), (us)
    text = PAREN_TAG_RE.sub(' ', text)
    # Keep unicode letters (\w includes accented chars), digits, spaces
    # Remove all other punctuation: &, -, ., ,, etc.
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove underscores (included in \w)
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def strip_legal_suffixes(name: str) -> str:
    """
    Remove legal entity suffixes from a cleaned (lowercased, no-punct) name.
    Strips from both ends since suffixes can appear at start or end.
    """
    tokens = name.split()
    # Strip from end (most common position)
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    # Strip from start (rare but occurs: "LLC Moncada Learning Center")
    while tokens and tokens[0] in LEGAL_SUFFIXES:
        tokens.pop(0)
    return ' '.join(tokens) if tokens else name  # fallback to original if all stripped


def normalize_name(name: str) -> str:
    """
    Full normalization for name: clean → strip legal suffixes.
    Used for token-overlap blocking and evaluation.
    """
    cleaned = clean_text(name)
    return strip_legal_suffixes(cleaned)


def normalize_name_light(name: str) -> str:
    """
    Light normalization for name: lowercase, collapse whitespace, keep accents.
    Used for character n-gram TF-IDF blocking.
    """
    return clean_text_light(name)


def normalize_address(address: str) -> str:
    """
    Normalize address: clean → expand abbreviations.
    """
    cleaned = clean_text(address)
    for pattern, replacement in _ADDR_ABBREV_COMPILED:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def normalize_address_light(address: str) -> str:
    """
    Light normalization for address: lowercase, collapse whitespace.
    """
    return clean_text_light(address)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing to a source dataframe.
    
    Adds columns:
        - norm_name: aggressively normalized name (for token blocking)
        - norm_name_light: lightly normalized name (for n-gram/embedding blocking)
        - norm_address: normalized address (for token/locality blocking)
        - norm_address_light: lightly normalized address (for embedding blocking)
        - norm_country: lowercased country
    """
    processed = df.copy()

    # Fill NAs
    for col in ['business_name', 'business_address', 'country']:
        processed[col] = processed[col].fillna('')

    # Normalized columns
    processed['norm_name'] = processed['business_name'].apply(normalize_name)
    processed['norm_name_light'] = processed['business_name'].apply(normalize_name_light)
    processed['norm_address'] = processed['business_address'].apply(normalize_address)
    processed['norm_address_light'] = processed['business_address'].apply(normalize_address_light)
    processed['norm_country'] = processed['country'].apply(
        lambda x: str(x).lower().strip() if pd.notna(x) else ''
    )

    return processed


def load_source(file_path: str) -> pd.DataFrame:
    """Load a source TSV and apply preprocessing."""
    df = pd.read_csv(file_path, sep='\t', dtype=str)
    return preprocess_dataframe(df)


def load_ground_truth(file_path: str) -> pd.DataFrame:
    """Load the ground truth TSV."""
    gt = pd.read_csv(file_path, sep='\t', dtype=str)
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    return gt


if __name__ == "__main__":
    import os
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "dataset"
    )

    # Test on a sample
    s1 = load_source(os.path.join(base, "train", "train_source1.tsv"))
    print(f"Loaded {len(s1):,} records from train_source1")
    print("\nSample preprocessing:")
    cols = ['entity_id', 'business_name', 'norm_name', 'norm_name_light', 
            'business_address', 'norm_address', 'norm_country']
    for _, row in s1.head(5).iterrows():
        for c in cols:
            print(f"  {c}: {row[c]}")
        print()

    # Spot-check on test France data
    test_s1 = load_source(os.path.join(base, "test", "test_source1.tsv"))
    france = test_s1[test_s1['norm_country'] == 'france'].head(5)
    print("=== France spot-check ===")
    for _, row in france.iterrows():
        for c in cols:
            print(f"  {c}: {row[c]}")
        print()
