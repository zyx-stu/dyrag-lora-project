"""
src/retrieval/build_index.py

Standalone script: builds the production FAISS index over the PubMed corpus.

Run once:
    python src/retrieval/build_index.py

This script:
  1. Downloads/loads the PubMed corpus via pubmed_corpus.py
  2. Embeds all documents with sentence-transformers
  3. Builds a FAISS flat index
  4. Saves index + documents to experiments/retrieval/pubmed_index/

After this runs once, all subsequent code loads the saved index
(FAISSRetriever.load()) instead of re-embedding — saving ~5 minutes
per run.

Estimated time: 5-10 minutes for 60K documents on RTX 5070 Ti
Estimated disk space: ~200MB (embeddings) + ~50MB (JSONL corpus)
"""

import sys
import time
import argparse
from pathlib import Path

# Make src importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.pubmed_corpus import build_corpus, load_corpus
from src.retrieval.faiss_retriever import FAISSRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORPUS_PATH = "experiments/retrieval/pubmed_corpus.jsonl"
INDEX_PATH  = "experiments/retrieval/pubmed_index"

RETRIEVER_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k":      3,
    "batch_size": 512,       # larger batch = faster embedding on GPU
    "normalize":  True,
    "device":     "cuda",
}


def main(force_rebuild: bool = False, max_docs: int = None):
    """
    Build and save the production FAISS index.

    Parameters
    ----------
    force_rebuild : re-download corpus and rebuild index even if they exist
    max_docs      : cap corpus size (useful for smoke tests; None = full corpus)
    """
    print("=" * 60)
    print("DyRAG-LoRA: Building Production FAISS Index")
    print("=" * 60)
    t_total = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load or build corpus
    # ------------------------------------------------------------------
    index_exists  = Path(INDEX_PATH).exists() and \
                    Path(f"{INDEX_PATH}/faiss.index").exists()
    corpus_exists = Path(CORPUS_PATH).exists()

    if index_exists and not force_rebuild:
        print(f"\nIndex already exists at {INDEX_PATH}")
        print("Use --force to rebuild. Loading existing index for verification...")
        _verify_existing_index()
        return

    print(f"\n[Step 1] Loading PubMed corpus...")
    if corpus_exists and not force_rebuild:
        corpus = load_corpus(CORPUS_PATH)
    else:
        corpus = build_corpus(force_rebuild=force_rebuild)

    if max_docs is not None:
        corpus = corpus[:max_docs]
        print(f"  Capped corpus at {len(corpus):,} documents (--max-docs)")

    print(f"  Corpus ready: {len(corpus):,} documents")

    # ------------------------------------------------------------------
    # Step 2: Build FAISS index
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Building FAISS index...")
    print(f"  Embedder: {RETRIEVER_CONFIG['model_name']}")
    print(f"  Batch size: {RETRIEVER_CONFIG['batch_size']}")
    print(f"  Device: {RETRIEVER_CONFIG['device']}")

    t_index = time.time()
    retriever = FAISSRetriever(config=RETRIEVER_CONFIG)
    retriever.build_index(corpus)
    index_time = time.time() - t_index

    print(f"  Index built in {index_time:.1f}s "
          f"({len(corpus)/index_time:.0f} docs/sec)")

    # ------------------------------------------------------------------
    # Step 3: Save index
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Saving index to {INDEX_PATH}...")
    retriever.save(INDEX_PATH)

    # ------------------------------------------------------------------
    # Step 4: Spot-check retrieval quality
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Spot-checking retrieval quality...")
    _spot_check(retriever)

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Index build complete in {total_time/60:.1f} minutes")
    print(f"Index location: {INDEX_PATH}")
    print(f"Documents indexed: {retriever.index.ntotal:,}")
    print(f"{'='*60}")


def _spot_check(retriever: FAISSRetriever) -> None:
    """
    Run 5 MedQA-style queries and print top retrieved document.
    This is a quick sanity check — not a formal evaluation.
    """
    test_queries = [
        "urinary tract infection treatment pregnancy antibiotic",
        "diabetes mellitus insulin type 2 metformin",
        "myocardial infarction ST elevation treatment thrombolysis",
        "meningitis bacterial CSF lumbar puncture diagnosis",
        "hypertension ACE inhibitor contraindication pregnancy",
    ]

    print(f"\n  Spot-check queries (top-1 result preview):")
    for q in test_queries:
        results = retriever.retrieve(q, k=1)
        if results:
            doc, score = results[0]
            print(f"\n  Q: {q}")
            print(f"  Score: {score:.3f} | Doc: {doc[:100]}...")
        else:
            print(f"\n  Q: {q} → No results (empty index?)")


def _verify_existing_index() -> None:
    """Load and spot-check an already-built index."""
    retriever = FAISSRetriever(config=RETRIEVER_CONFIG)
    retriever.load(INDEX_PATH)
    print(f"  Loaded: {retriever.index.ntotal:,} vectors, dim={retriever.dim}")
    _spot_check(retriever)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build production FAISS index over PubMed abstracts"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild even if index already exists"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Cap corpus size (e.g. 5000 for smoke test)"
    )
    args = parser.parse_args()

    main(force_rebuild=args.force, max_docs=args.max_docs)