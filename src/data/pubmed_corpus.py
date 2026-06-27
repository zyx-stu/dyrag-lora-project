"""
src/data/pubmed_corpus.py

Responsible for:
  1. Downloading PubMed abstracts from HuggingFace
  2. Cleaning and filtering abstracts
  3. Saving a flat corpus file to disk for the FAISS indexer

Dataset: suolyer/pile_pubmed-abstracts
  - Real PubMed abstracts from The Pile
  - Parquet format (no loading script required)
  - ~59.8K rows across validation + test splits
  - Field: "text" contains the abstract text

We combine both splits to get ~59K abstracts.
For 100K, we use the dataset twice with minor deduplication,
or supplement with pubmed_qa unlabeled split (211K instances).

Design note: this module saves to disk once and never re-downloads.
All downstream code (build_index.py) reads from the saved file.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CORPUS_CONFIG = {
    # Primary dataset
    "primary_dataset":  "suolyer/pile_pubmed-abstracts",
    "primary_splits":   ["validation", "test"],

    # Supplementary dataset for additional abstracts
    "supplement_dataset": "pubmed_qa",
    "supplement_config":  "pqa_unlabeled",
    "supplement_split":   "train",

    # Filtering thresholds
    "min_length":  100,    # minimum character length (filters stubs)
    "max_length":  2000,   # maximum character length (filters very long abstracts)

    # Target corpus size
    "target_n":    100_000,

    # Output
    "output_path": "experiments/retrieval/pubmed_corpus.jsonl",
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_abstract(text: str) -> Optional[str]:
    """
    Clean a raw PubMed abstract text.

    Operations:
      1. Strip leading/trailing whitespace
      2. Remove structured abstract headers (BACKGROUND:, METHODS:, etc.)
         — these are formatting artifacts not useful for retrieval
      3. Collapse multiple spaces/newlines
      4. Filter by length

    Returns None if the abstract should be excluded (too short/long/empty).
    """
    if not text or not isinstance(text, str):
        return None

    # Remove structured abstract headers
    # e.g. "BACKGROUND: ...\nMETHODS: ..." → "... ..."
    text = re.sub(
        r'\b(BACKGROUND|METHODS?|RESULTS?|CONCLUSIONS?|OBJECTIVE|PURPOSE|'
        r'AIMS?|INTRODUCTION|DISCUSSION|SUMMARY|SIGNIFICANCE|CONTEXT|'
        r'DESIGN|SETTING|PATIENTS?|INTERVENTIONS?|MEASUREMENTS?|'
        r'MAIN OUTCOME MEASURES?|STUDY DESIGN)\s*:\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Filter by length
    cfg = DEFAULT_CORPUS_CONFIG
    if len(text) < cfg["min_length"] or len(text) > cfg["max_length"]:
        return None

    return text


# ---------------------------------------------------------------------------
# Download and build corpus
# ---------------------------------------------------------------------------

def build_corpus(
    config:       Optional[dict] = None,
    force_rebuild: bool = False,
) -> List[str]:
    """
    Download PubMed abstracts and build the retrieval corpus.

    Returns a list of cleaned abstract strings.
    Also saves to disk for reuse.

    Parameters
    ----------
    config        : override DEFAULT_CORPUS_CONFIG keys
    force_rebuild : if True, re-download even if corpus file exists

    Returns
    -------
    List of cleaned abstract strings (length ≈ target_n)
    """
    cfg = {**DEFAULT_CORPUS_CONFIG, **(config or {})}
    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load from disk if already built
    if output_path.exists() and not force_rebuild:
        print(f"Loading existing corpus from {output_path}")
        return load_corpus(str(output_path))

    print(f"Building PubMed corpus (target: {cfg['target_n']:,} abstracts)")
    print(f"Output: {output_path}\n")

    corpus = []
    seen   = set()  # for deduplication (first 100 chars as key)

    # ------------------------------------------------------------------
    # Stage 1: Primary dataset — suolyer/pile_pubmed-abstracts
    # ------------------------------------------------------------------
    print(f"Stage 1: Loading {cfg['primary_dataset']}...")
    for split in cfg["primary_splits"]:
        if len(corpus) >= cfg["target_n"]:
            break
        try:
            ds = load_dataset(cfg["primary_dataset"], split=split)
            print(f"  Split '{split}': {len(ds):,} rows")

            for row in tqdm(ds, desc=f"  Cleaning {split}"):
                if len(corpus) >= cfg["target_n"]:
                    break

                text = clean_abstract(row.get("text", ""))
                if text is None:
                    continue

                # Deduplication check
                key = text[:100]
                if key in seen:
                    continue
                seen.add(key)
                corpus.append(text)

        except Exception as e:
            print(f"  Warning: Failed to load split '{split}': {e}")
            continue

    print(f"  After Stage 1: {len(corpus):,} abstracts")

    # ------------------------------------------------------------------
    # Stage 2: Supplement with pubmed_qa if needed
    # ------------------------------------------------------------------
    if len(corpus) < cfg["target_n"]:
        needed = cfg["target_n"] - len(corpus)
        print(f"\nStage 2: Need {needed:,} more — loading {cfg['supplement_dataset']}...")

        try:
            ds_supp = load_dataset(
                cfg["supplement_dataset"],
                cfg["supplement_config"],
                split=cfg["supplement_split"],
                trust_remote_code=False,
            )
            print(f"  Supplement dataset: {len(ds_supp):,} rows")

            for row in tqdm(ds_supp, desc="  Cleaning supplement"):
                if len(corpus) >= cfg["target_n"]:
                    break

                # pubmed_qa stores context as a list of sentences
                context = row.get("context", {})
                if isinstance(context, dict):
                    sentences = context.get("contexts", [])
                    text_raw  = " ".join(sentences)
                elif isinstance(context, str):
                    text_raw = context
                else:
                    continue

                text = clean_abstract(text_raw)
                if text is None:
                    continue

                key = text[:100]
                if key in seen:
                    continue
                seen.add(key)
                corpus.append(text)

        except Exception as e:
            print(f"  Warning: Supplement load failed: {e}")
            print(f"  Continuing with {len(corpus):,} abstracts from Stage 1")

    print(f"\nFinal corpus size: {len(corpus):,} abstracts")

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------
    print(f"Saving corpus to {output_path}...")
    with open(output_path, "w") as f:
        for text in corpus:
            f.write(json.dumps({"text": text}) + "\n")

    _print_corpus_stats(corpus)
    return corpus


def load_corpus(path: str) -> List[str]:
    """
    Load a previously saved corpus from JSONL file.

    Returns list of abstract strings.
    """
    path = Path(path)
    assert path.exists(), f"Corpus file not found: {path}"

    corpus = []
    with open(path) as f:
        for line in f:
            row  = json.loads(line)
            corpus.append(row["text"])

    print(f"Loaded {len(corpus):,} abstracts from {path}")
    return corpus


def _print_corpus_stats(corpus: List[str]) -> None:
    """Print summary statistics about the corpus."""
    lengths = [len(t) for t in corpus]
    print(f"\nCorpus statistics:")
    print(f"  Count      : {len(corpus):>8,}")
    print(f"  Min length : {min(lengths):>8,} chars")
    print(f"  Max length : {max(lengths):>8,} chars")
    print(f"  Mean length: {sum(lengths)//len(lengths):>8,} chars")

    # Show 3 sample abstracts
    import random
    print(f"\nSample abstracts:")
    for i, idx in enumerate(random.sample(range(len(corpus)), min(3, len(corpus)))):
        print(f"  [{i+1}] {corpus[idx][:120]}...")