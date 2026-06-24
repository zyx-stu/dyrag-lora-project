"""
src/retrieval/faiss_retriever.py

Responsible for:
  1. Embedding a corpus of documents using a sentence-transformer
  2. Building a FAISS index for fast approximate nearest-neighbour search
  3. Retrieving top-k documents for a given query at inference/training time
  4. Saving and loading the index to avoid recomputing embeddings

This is Phase 0 of DyRAG-LoRA: a frozen retriever using a pre-trained
sentence-transformer (no fine-tuning yet). The embedder is fine-tuned
in Phases 1-3 using the loss-delta signal.

Architecture:
  Query string
      │
      ▼
  SentenceTransformer.encode()   [384-dim embedding for all-MiniLM-L6-v2]
      │
      ▼
  FAISS IndexFlatIP              [exact inner product search]
      │
      ▼
  Top-k document indices
      │
      ▼
  Retrieved document strings

Why IndexFlatIP (inner product) instead of IndexFlatL2 (L2 distance)?
  sentence-transformers normalizes embeddings to unit length by default.
  For unit vectors: cosine_similarity = inner_product.
  IndexFlatIP on normalized vectors = cosine similarity search.
  This is the standard setup for semantic retrieval.

Why all-MiniLM-L6-v2 as the base embedder?
  - 384-dim embeddings (vs 768 for larger models) → faster indexing
  - 22M parameters → fits comfortably alongside LLaMA in VRAM
  - Strong out-of-the-box performance on semantic similarity benchmarks
  - Widely used in RAG literature → fair comparison baseline
  The loss-delta signal will fine-tune this embedder to improve
  domain-specific retrieval precision in later phases.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_RETRIEVER_CONFIG = {
    "model_name":   "sentence-transformers/all-MiniLM-L6-v2",
    "top_k":        3,
    "batch_size":   256,     # embedding batch size (not training batch size)
    "normalize":    True,    # normalize embeddings for cosine similarity
    "device":       "cuda",  # embed on GPU for speed
}


class FAISSRetriever:
    """
    Dense retriever backed by a FAISS flat index.

    Workflow:
      1. build_index(documents)   — embed all docs, build FAISS index
      2. retrieve(query, k)       — embed query, search index, return top-k
      3. save(path)               — persist index + documents to disk
      4. load(path)               — restore from disk (skip re-embedding)

    Parameters
    ----------
    config : dict
        Override any key in DEFAULT_RETRIEVER_CONFIG.
        Most important: "model_name" for ablations with different embedders,
        "top_k" for the k ablation study in Week 8.
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = {**DEFAULT_RETRIEVER_CONFIG, **(config or {})}

        print(f"Loading embedder: {self.cfg['model_name']}")
        self.embedder = SentenceTransformer(
            self.cfg["model_name"],
            device=self.cfg["device"],
        )

        self.index:     Optional[faiss.Index] = None
        self.documents: Optional[List[str]]   = None
        self.dim:       Optional[int]         = None

        logger.info(f"FAISSRetriever initialized: {self.cfg['model_name']}")

    def build_index(self, documents: List[str]) -> None:
        """
        Embed all documents and build a FAISS flat index.

        Parameters
        ----------
        documents : list of strings
            The full knowledge base. For DyRAG-LoRA Phase 0, this is
            100K PubMed abstracts. For smoke tests, pass a small list.

        Notes on FAISS index type choice:
          IndexFlatIP: exact search, no approximation error.
          For 100K documents at 384-dim, exact search is fast enough
          (<50ms per query on GPU) and avoids the accuracy cost of
          approximate methods (IVF, HNSW). If the corpus grows to
          1M+, switch to IndexIVFFlat with nlist~1000.
        """
        self.documents = documents
        n_docs         = len(documents)

        print(f"Embedding {n_docs} documents (batch_size={self.cfg['batch_size']})...")

        # Embed in batches to avoid OOM on large corpora
        embeddings = self.embedder.encode(
            documents,
            batch_size       = self.cfg["batch_size"],
            normalize_embeddings = self.cfg["normalize"],
            show_progress_bar    = True,
            convert_to_numpy     = True,
        )

        # embeddings shape: [n_docs, embedding_dim]
        self.dim = embeddings.shape[1]
        print(f"Embeddings shape: {embeddings.shape}  (dim={self.dim})")

        # Build FAISS index
        # IndexFlatIP: brute-force inner product search
        # No training required (unlike IVF variants)
        self.index = faiss.IndexFlatIP(self.dim)

        # Move index to GPU for faster search
        # faiss.StandardGpuResources() allocates a GPU memory pool
        
        print("FAISS index built (CPU search)")

        # Add embeddings to index
        # FAISS expects float32, ensure correct dtype
        self.index.add(embeddings.astype(np.float32))
        print(f"FAISS index built: {self.index.ntotal} vectors indexed")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a query string.

        Parameters
        ----------
        query : str
            The question or search string.
        k : int, optional
            Number of documents to retrieve. Defaults to config top_k.

        Returns
        -------
        List of (document_text, similarity_score) tuples,
        sorted by similarity descending (most relevant first).
        """
        assert self.index is not None, "Call build_index() before retrieve()"

        k = k or self.cfg["top_k"]

        # Embed the query — single string, returns [1, dim] array
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings = self.cfg["normalize"],
            convert_to_numpy     = True,
        ).astype(np.float32)

        # Search the index
        # scores: [1, k] — inner product similarities (= cosine for normalized vecs)
        # indices: [1, k] — indices into self.documents
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for empty slots (shouldn't happen with FlatIP)
                continue
            results.append((self.documents[idx], float(score)))

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        k: Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Retrieve top-k documents for a batch of queries.

        Used in the loss-delta computation (Week 3) where we process
        multiple training examples per batch.

        Returns list of lists — one result list per query.
        """
        assert self.index is not None, "Call build_index() before retrieve_batch()"

        k = k or self.cfg["top_k"]

        query_embeddings = self.embedder.encode(
            queries,
            normalize_embeddings = self.cfg["normalize"],
            convert_to_numpy     = True,
            batch_size           = min(len(queries), 64),
        ).astype(np.float32)

        scores_batch, indices_batch = self.index.search(query_embeddings, k)

        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                if idx != -1:
                    results.append((self.documents[idx], float(score)))
            all_results.append(results)

        return all_results

    def format_retrieved_context(
        self,
        results: List[Tuple[str, float]],
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Format retrieved (doc, score) tuples into a context string
        for injection into the LLM prompt.

        Example output:
            [Document 1]
            Nitrofurantoin is a nitrofuran antibiotic used for UTI...

            ---

            [Document 2]
            Doxycycline is contraindicated in pregnancy due to...
        """
        parts = []
        for i, (doc, score) in enumerate(results, 1):
            parts.append(f"[Document {i}]\n{doc}")
        return separator.join(parts)

    def save(self, save_dir: str) -> None:
        """
        Save FAISS index and document list to disk.

        Saves two files:
          {save_dir}/faiss.index   — the FAISS index (binary)
          {save_dir}/documents.json — the document corpus

        This avoids re-embedding on every run.
        For 100K PubMed abstracts, embedding takes ~5 minutes —
        saving to disk and loading takes ~10 seconds.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Must move GPU index to CPU before saving
        
        cpu_index = self.index

        faiss.write_index(cpu_index, str(save_path / "faiss.index"))

        with open(save_path / "documents.json", "w") as f:
            json.dump(self.documents, f)

        config_to_save = {**self.cfg, "dim": self.dim, "n_docs": len(self.documents)}
        with open(save_path / "retriever_config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)

        print(f"Retriever saved to: {save_dir}")
        print(f"  Index: {self.index.ntotal} vectors")
        print(f"  Documents: {len(self.documents)}")

    def load(self, save_dir: str) -> None:
        """
        Load a previously saved FAISS index and document list from disk.
        """
        save_path = Path(save_dir)
        assert save_path.exists(), f"Save directory not found: {save_dir}"

        print(f"Loading retriever from: {save_dir}")

        # Load CPU index then move to GPU
        cpu_index  = faiss.read_index(str(save_path / "faiss.index"))
        
        self.index = cpu_index

        with open(save_path / "documents.json") as f:
            self.documents = json.load(f)

        with open(save_path / "retriever_config.json") as f:
            saved_cfg  = json.load(f)
            self.dim   = saved_cfg["dim"]

        print(f"Retriever loaded: {self.index.ntotal} vectors, dim={self.dim}")
