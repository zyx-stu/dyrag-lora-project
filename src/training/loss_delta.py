"""
src/training/loss_delta.py

The core novelty of DyRAG-LoRA.

Computes the loss-delta signal: for each retrieved document dᵢ,
how much does prepending it to the context reduce the model's
cross-entropy loss on the correct answer?

    δᵢ = L(q, a) - L(q + dᵢ, a)

    δᵢ > τ  → document is HELPFUL  (positive example for embedder)
    δᵢ < -τ → document is HARMFUL  (negative example for embedder)
    |δᵢ|≤ τ → document is NEUTRAL  (excluded from training signal)

This module also constructs (query, positive, negative) triplets
for fine-tuning the retrieval embedder via MultipleNegativesRankingLoss.

Public API:
    LossDeltaComputer.compute_for_example()  — single example
    LossDeltaComputer.compute_for_dataset()  — batch over dataset
    build_triplets()                          — triplet construction
    gate1_sanity_check()                      — Gate 1 validation
"""
# tau operating regime:
#   tau=0.05 is appropriate for mid-difficulty questions (baseline loss 1.5-2.5)
#   For easy questions (baseline loss <1.5), model has strong parametric knowledge
#   and context interference dominates — most deltas will be negative.
#   For very hard questions (baseline loss >3.0), most deltas will be positive
#   but less discriminative. The signal is most useful in the 1.5-2.5 range.

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentDelta:
    """
    Loss-delta result for a single (query, document) pair.

    Fields
    ------
    document    : the retrieved document text
    loss_without: L(q, a) — baseline loss without this document
    loss_with   : L(q + dᵢ, a) — loss with document in context
    delta       : loss_without - loss_with  (positive = helpful)
    label       : "positive", "negative", or "neutral"
    """
    document:     str
    loss_without: float
    loss_with:    float
    delta:        float
    label:        str   # "positive" | "negative" | "neutral"


@dataclass
class ExampleDeltaResult:
    """
    Loss-delta results for all retrieved documents for one training example.
    """
    question:    str
    answer:      str
    answer_idx:  str
    documents:   List[DocumentDelta]
    # Convenience accessors
    positives:   List[str] = field(default_factory=list)  # helpful doc texts
    negatives:   List[str] = field(default_factory=list)  # harmful doc texts

    def __post_init__(self):
        self.positives = [d.document for d in self.documents if d.label == "positive"]
        self.negatives = [d.document for d in self.documents if d.label == "negative"]


@dataclass
class Triplet:
    """
    A single training triplet for the retrieval embedder.
    (query, positive_doc, negative_doc)
    """
    query:    str
    positive: str
    negative: str


# ---------------------------------------------------------------------------
# Core loss computation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical expert answering USMLE-style questions. "
    "Read the question carefully, consider all options, and provide the correct answer."
)


def _format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in options.items())


def _build_prompt_with_context(
    tokenizer:   AutoTokenizer,
    question:    str,
    options:     Dict[str, str],
    answer:      str,
    answer_idx:  str,
    context_doc: Optional[str] = None,
) -> str:
    """
    Build a full training prompt, optionally prepending a retrieved document.

    With context_doc:
        [Retrieved Context]
        {document text}

        Question: {question}
        Options: ...

    Without context_doc:
        Question: {question}
        Options: ...

    The context is prepended to the user turn, not as a separate message,
    to keep the format consistent with the training data in medqa_dataset.py.
    """
    options_str = _format_options(options)

    if context_doc is not None:
        user_content = (
            f"[Retrieved Context]\n{context_doc}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_str}"
        )
    else:
        user_content = (
            f"Question: {question}\n\n"
            f"Options:\n{options_str}"
        )

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": f"The answer is {answer_idx}: {answer}"},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _compute_loss_on_answer(
    model:       torch.nn.Module,
    tokenizer:   AutoTokenizer,
    full_prompt: str,
    max_length:  int = 640,
) -> float:
    """
    Compute cross-entropy loss on the answer tokens only.

    This is the core operation of the loss-delta signal.
    Uses torch.no_grad() — we are MEASURING loss, not training.

    The loss masking here mirrors medqa_dataset.py:
    prompt tokens → label=-100 (ignored)
    answer tokens → label=token_id (supervised)

    Parameters
    ----------
    model       : the QLoRA model (frozen base + LoRA adapters)
    tokenizer   : matching tokenizer
    full_prompt : complete formatted string including question + answer
    max_length  : safety cap to prevent OOM on long contexts

    Returns
    -------
    float : mean cross-entropy loss over answer tokens
            Lower = model is more confident about the correct answer
    """
    device = next(model.parameters()).device

    # Tokenize the full prompt
    tokenized = tokenizer(
        full_prompt,
        return_tensors  = "pt",
        max_length      = max_length,
        truncation      = True,
        padding         = False,
    )
    input_ids      = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # Build labels — mask prompt, supervise answer tokens
    # We find the answer boundary by tokenizing the prompt-only version
    # (same approach as medqa_dataset.py for consistency)
    prompt_only = _build_prompt_with_context(
        tokenizer   = tokenizer,
        question    = "",         # placeholder — we only need length
        options     = {},
        answer      = "",
        answer_idx  = "",
        context_doc = None,
    )
    # More robust: find where "<|start_header_id|>assistant" begins
    # and count tokens up to that point
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    marker_tokens = tokenizer(
        assistant_marker, add_special_tokens=False
    )["input_ids"]
    marker_len = len(marker_tokens)

    # Find the assistant header position in input_ids
    input_list = input_ids[0].tolist()
    prompt_len = len(input_list)  # fallback: mask everything

    for i in range(len(input_list) - marker_len):
        if input_list[i:i+marker_len] == marker_tokens:
            # +marker_len to skip past the header itself
            prompt_len = i + marker_len
            break

    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    # Safety: if no answer tokens remain after truncation, skip
    if (labels != -100).sum() == 0:
        return float("nan")

    # Forward pass — inference only, no gradient computation
    # This is safe because:
    # 1. We only need the scalar loss value, not gradients
    # 2. torch.no_grad() prevents gradient graph construction → saves VRAM
    # 3. The LoRA adapter weights are not updated here — only in trainer.py
    with torch.no_grad():
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )

    return outputs.loss.item()


# ---------------------------------------------------------------------------
# Main LossDeltaComputer class
# ---------------------------------------------------------------------------

class LossDeltaComputer:
    """
    Computes loss-delta values for retrieved documents.

    Parameters
    ----------
    model      : QLoRA model — used for loss computation (inference only)
    tokenizer  : matching tokenizer
    tau        : threshold for positive/negative labeling (default 0.05)
    max_length : maximum sequence length for loss computation
    """

    def __init__(
        self,
        model:      torch.nn.Module,
        tokenizer:  AutoTokenizer,
        tau:        float = 0.05,
        max_length: int   = 640,
    ):
        self.model      = model
        self.tokenizer  = tokenizer
        self.tau        = tau
        self.max_length = max_length

        # Put model in eval mode for loss-delta computation
        # (disables dropout — important for reproducible loss values)
        self.model.eval()
        logger.info(f"LossDeltaComputer initialized: tau={tau}")

    def compute_for_example(
        self,
        question:    str,
        options:     Dict[str, str],
        answer:      str,
        answer_idx:  str,
        documents:   List[str],
    ) -> ExampleDeltaResult:
        """
        Compute loss-delta for all retrieved documents for one example.

        Algorithm:
          1. Compute baseline loss L_without (no context)
          2. For each document dᵢ:
               Compute L_with = loss with dᵢ prepended to context
               δᵢ = L_without - L_with
               Label dᵢ based on δᵢ vs tau
          3. Return ExampleDeltaResult with all deltas and labels

        Parameters
        ----------
        question, options, answer, answer_idx : from MedQA dataset row
        documents : list of retrieved document strings (typically k=3)

        Returns
        -------
        ExampleDeltaResult with DocumentDelta for each document
        """
        # ------------------------------------------------------------------
        # Step 1: Baseline loss — no retrieved context
        # ------------------------------------------------------------------
        prompt_without = _build_prompt_with_context(
            tokenizer   = self.tokenizer,
            question    = question,
            options     = options,
            answer      = answer,
            answer_idx  = answer_idx,
            context_doc = None,
        )
        loss_without = _compute_loss_on_answer(
            self.model, self.tokenizer, prompt_without, self.max_length
        )

        # ------------------------------------------------------------------
        # Step 2: Loss with each document
        # ------------------------------------------------------------------
        doc_deltas = []
        for doc in documents:
            prompt_with = _build_prompt_with_context(
                tokenizer   = self.tokenizer,
                question    = question,
                options     = options,
                answer      = answer,
                answer_idx  = answer_idx,
                context_doc = doc,
            )
            loss_with = _compute_loss_on_answer(
                self.model, self.tokenizer, prompt_with, self.max_length
            )

            # Skip NaN results (truncated sequences)
            if loss_with != loss_with:  # NaN check
                continue

            delta = loss_without - loss_with

            # Label based on threshold
            if delta > self.tau:
                label = "positive"
            elif delta < -self.tau:
                label = "negative"
            else:
                label = "neutral"

            doc_deltas.append(DocumentDelta(
                document     = doc,
                loss_without = loss_without,
                loss_with    = loss_with,
                delta        = delta,
                label        = label,
            ))

        return ExampleDeltaResult(
            question   = question,
            answer     = answer,
            answer_idx = answer_idx,
            documents  = doc_deltas,
        )

    def compute_for_dataset(
        self,
        dataset,
        retriever,
        n_examples:  int  = 500,
        save_path:   Optional[str] = None,
    ) -> List[ExampleDeltaResult]:
        """
        Compute loss-delta for n_examples from the dataset.

        This is the main entry point for Phase 0 → Phase 1 transition.

        Parameters
        ----------
        dataset    : HuggingFace dataset (GBaker/MedQA-USMLE-4-options)
        retriever  : FAISSRetriever instance with index already built
        n_examples : number of examples to process (500 per phase in plan)
        save_path  : if set, save results to JSONL for inspection

        Returns
        -------
        List of ExampleDeltaResult, one per processed example
        """
        results = []
        n = min(n_examples, len(dataset))

        print(f"\nComputing loss-delta for {n} examples...")
        print(f"Tau threshold: {self.tau}")
        print(f"Documents per example: k={retriever.cfg['top_k']}\n")

        pos_count = neg_count = neu_count = 0

        for i in tqdm(range(n), desc="Loss-delta"):
            row = dataset[i]

            # Retrieve documents for this query
            retrieved = retriever.retrieve(row["question"])
            docs = [doc for doc, _ in retrieved]

            # Compute deltas
            result = self.compute_for_example(
                question   = row["question"],
                options    = row["options"],
                answer     = row["answer"],
                answer_idx = row["answer_idx"],
                documents  = docs,
            )
            results.append(result)

            # Running counts
            for d in result.documents:
                if d.label == "positive":  pos_count += 1
                elif d.label == "negative": neg_count += 1
                else:                       neu_count += 1

            # Progress summary every 50 examples
            if (i + 1) % 50 == 0:
                total_docs = pos_count + neg_count + neu_count
                print(
                    f"  [{i+1}/{n}] "
                    f"pos={pos_count} ({100*pos_count/total_docs:.0f}%) | "
                    f"neg={neg_count} ({100*neg_count/total_docs:.0f}%) | "
                    f"neu={neu_count} ({100*neu_count/total_docs:.0f}%)"
                )

        # Save to JSONL for manual inspection (Gate 1)
        if save_path:
            _save_results(results, save_path)
            print(f"\nResults saved to: {save_path}")

        # Summary
        total = pos_count + neg_count + neu_count
        print(f"\nLoss-delta summary:")
        print(f"  Positive (helpful) : {pos_count:4d} ({100*pos_count/total:.1f}%)")
        print(f"  Negative (harmful) : {neg_count:4d} ({100*neg_count/total:.1f}%)")
        print(f"  Neutral            : {neu_count:4d} ({100*neu_count/total:.1f}%)")
        print(f"  Total documents    : {total}")

        return results


# ---------------------------------------------------------------------------
# Triplet construction
# ---------------------------------------------------------------------------

def build_triplets(
    delta_results:   List[ExampleDeltaResult],
    corpus:          List[str],
    distractor_ratio: float = 0.5,
    max_triplets:    Optional[int] = None,
) -> List[Triplet]:
    """
    Construct (query, positive, negative) triplets from loss-delta results.

    Strategy:
      For each example with at least one positive document:
        For each positive document d⁺:
          If a negative document d⁻ exists: use it (hard negative)
          Else: sample a random document from corpus (easy negative)

    The distractor_ratio controls what fraction of negatives are
    randomly sampled from the corpus vs. naturally retrieved negatives.
    Lower distractor_ratio = more naturally retrieved negatives = harder.

    Parameters
    ----------
    delta_results    : output of LossDeltaComputer.compute_for_dataset()
    corpus           : the full document corpus (for random negative sampling)
    distractor_ratio : fraction of negatives to sample randomly
    max_triplets     : cap on total triplets (None = no cap)

    Returns
    -------
    List of Triplet objects ready for embedder fine-tuning
    """
    triplets = []

    for result in delta_results:
        if not result.positives:
            continue   # no positive docs → no triplet for this example

        for pos_doc in result.positives:
            # Choose negative: use retrieved negative if available,
            # otherwise sample from corpus
            if result.negatives and random.random() > distractor_ratio:
                neg_doc = random.choice(result.negatives)
            else:
                # Random corpus document (ensure it's not the positive)
                neg_doc = random.choice(corpus)
                while neg_doc == pos_doc:
                    neg_doc = random.choice(corpus)

            triplets.append(Triplet(
                query    = result.question,
                positive = pos_doc,
                negative = neg_doc,
            ))

    if max_triplets:
        random.shuffle(triplets)
        triplets = triplets[:max_triplets]

    return triplets


# ---------------------------------------------------------------------------
# Gate 1 sanity check
# ---------------------------------------------------------------------------

def gate1_sanity_check(
    delta_results: List[ExampleDeltaResult],
    n_inspect:     int = 20,
) -> dict:
    """
    Gate 1: Manual sanity check on loss-delta labels.

    Prints the first n_inspect examples with their documents,
    delta values, and labels for manual review.

    Pass condition: at least 15/20 examples have at least one
    positive document and the labels make intuitive sense.

    Returns a summary dict with pass/fail recommendation.
    """
    print("\n" + "=" * 70)
    print("GATE 1 SANITY CHECK — Manual inspection of loss-delta labels")
    print("=" * 70)
    print(f"Inspecting {n_inspect} examples...")
    print("For each example, check: does the 'positive' document")
    print("actually contain information relevant to the correct answer?\n")

    examples_with_positives = 0
    inspect = delta_results[:n_inspect]

    for i, result in enumerate(inspect):
        has_positive = len(result.positives) > 0
        if has_positive:
            examples_with_positives += 1

        print(f"─── Example {i+1}/{n_inspect} ───")
        print(f"Q: {result.question[:100]}...")
        print(f"A: {result.answer_idx}: {result.answer}")
        print()

        for j, doc_delta in enumerate(result.documents):
            marker = "✓" if doc_delta.label == "positive" else (
                     "✗" if doc_delta.label == "negative" else "~")
            print(f"  Doc {j+1} [{marker} {doc_delta.label.upper():8s}] "
                  f"δ={doc_delta.delta:+.4f} "
                  f"(L_without={doc_delta.loss_without:.4f}, "
                  f"L_with={doc_delta.loss_with:.4f})")
            print(f"  {doc_delta.document[:120]}...")
        print()

    # Gate 1 verdict
    pass_rate = examples_with_positives / n_inspect
    passed    = pass_rate >= 0.75  # 15/20 = 75%

    print("=" * 70)
    print(f"Gate 1 Result: {examples_with_positives}/{n_inspect} examples "
          f"have at least one positive document ({100*pass_rate:.0f}%)")
    print(f"Pass threshold: 15/20 (75%)")
    print(f"Gate 1: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 70)

    if not passed:
        print("\nDIAGNOSIS STEPS if Gate 1 fails:")
        print("  1. Check tau — if too high, too few positives. Try tau=0.02")
        print("  2. Check retriever — are retrieved docs topically relevant?")
        print("  3. Check prompt format — is context correctly prepended?")
        print("  4. Check loss masking — are answer tokens getting gradients?")

    return {
        "passed":                   passed,
        "examples_with_positives":  examples_with_positives,
        "n_inspected":              n_inspect,
        "pass_rate":                pass_rate,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_results(results: List[ExampleDeltaResult], path: str) -> None:
    """Save loss-delta results to JSONL for manual inspection."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            entry = {
                "question":   r.question,
                "answer":     r.answer,
                "answer_idx": r.answer_idx,
                "documents":  [
                    {
                        "document":     d.document[:200],
                        "loss_without": round(d.loss_without, 4),
                        "loss_with":    round(d.loss_with, 4),
                        "delta":        round(d.delta, 4),
                        "label":        d.label,
                    }
                    for d in r.documents
                ],
                "n_positive": len(r.positives),
                "n_negative": len(r.negatives),
            }
            f.write(json.dumps(entry) + "\n")
