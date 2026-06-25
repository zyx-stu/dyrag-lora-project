"""
tests/gate1_validation.py

GATE 1 — Real validation of the loss-delta signal on 20 MedQA examples.

Uses actual MedQA training data as the retrieval corpus (proxy for PubMed).
This is the gate decision that must be recorded before Week 4 proceeds.

Pass condition: ≥15/20 examples produce at least one positive document.
If this fails, diagnose before proceeding — the entire DyRAG-LoRA
contribution depends on this signal being meaningful.
"""

import sys, json
sys.path.insert(0, '/home/user/dyrag-lora')

import torch
from datasets import load_dataset
from src.training.model_loader import load_model_and_tokenizer
from src.training.loss_delta import LossDeltaComputer, gate1_sanity_check
from src.retrieval.faiss_retriever import FAISSRetriever

print("=" * 70)
print("GATE 1 — Real Loss-Delta Validation on MedQA")
print("=" * 70)

# ---------------------------------------------------------------
# Load model
# ---------------------------------------------------------------
print("\n[Setup 1] Loading model...")
model, tokenizer = load_model_and_tokenizer()
model.eval()

# ---------------------------------------------------------------
# Load dataset — use train split for corpus, test split for eval
# ---------------------------------------------------------------
print("[Setup 2] Loading MedQA dataset...")
ds_train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
ds_test  = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

# Build corpus from training question stems (proxy for PubMed)
# We use question text + correct answer as document content
# This gives rich, medically grounded documents for retrieval
print("[Setup 3] Building retrieval corpus from MedQA training questions...")
corpus = []
for row in ds_train.select(range(2000)):   # 2000 docs for fast indexing
    doc = (
        f"{row['question']} "
        f"The correct answer is {row['answer_idx']}: {row['answer']}."
    )
    corpus.append(doc)
print(f"  Corpus size: {len(corpus)} documents")

# ---------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------
print("[Setup 4] Building FAISS index...")
retriever = FAISSRetriever(config={"top_k": 3, "device": "cuda"})
retriever.build_index(corpus)

# ---------------------------------------------------------------
# Select 20 test examples — stratified across difficulty
# Use examples where baseline loss is likely >1.5 (harder questions)
# We sample from positions 50-250 to avoid the very easiest examples
# ---------------------------------------------------------------
print("[Setup 5] Selecting 20 test examples...")
test_examples = ds_test.select(range(50, 70))  # 20 examples

# ---------------------------------------------------------------
# Compute loss-delta
# ---------------------------------------------------------------
print("\n[Gate 1] Computing loss-delta on 20 real MedQA examples...")
print("This will take approximately 5-8 minutes...\n")

computer = LossDeltaComputer(model, tokenizer, tau=0.05)

results = []
baseline_losses = []

for i, row in enumerate(test_examples):
    # Retrieve documents
    retrieved = retriever.retrieve(row["question"], k=3)
    docs = [doc for doc, _ in retrieved]

    # Compute loss-delta
    result = computer.compute_for_example(
        question   = row["question"],
        options    = row["options"],
        answer     = row["answer"],
        answer_idx = row["answer_idx"],
        documents  = docs,
    )
    results.append(result)

    # Track baseline loss for difficulty analysis
    if result.documents:
        baseline_losses.append(result.documents[0].loss_without)

    # Progress
    n_pos = len(result.positives)
    n_neg = len(result.negatives)
    baseline = result.documents[0].loss_without if result.documents else 0
    print(f"  [{i+1:2d}/20] loss={baseline:.3f} | "
          f"pos={n_pos} neg={n_neg} "
          f"neu={len(result.documents)-n_pos-n_neg} | "
          f"Q: {row['question'][:55]}...")

# ---------------------------------------------------------------
# Gate 1 verdict
# ---------------------------------------------------------------
gate_result = gate1_sanity_check(results, n_inspect=20)

# ---------------------------------------------------------------
# Additional analysis for paper
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("ADDITIONAL ANALYSIS FOR PAPER")
print("=" * 70)

if baseline_losses:
    avg_baseline = sum(baseline_losses) / len(baseline_losses)
    easy   = sum(1 for l in baseline_losses if l < 1.5)
    medium = sum(1 for l in baseline_losses if 1.5 <= l < 2.5)
    hard   = sum(1 for l in baseline_losses if l >= 2.5)
    print(f"\nBaseline loss distribution (n={len(baseline_losses)}):")
    print(f"  Mean baseline loss : {avg_baseline:.4f}")
    print(f"  Easy   (<1.5)      : {easy:2d} examples")
    print(f"  Medium (1.5-2.5)   : {medium:2d} examples")
    print(f"  Hard   (>2.5)      : {hard:2d} examples")

# Label distribution
all_deltas = [d for r in results for d in r.documents]
pos = sum(1 for d in all_deltas if d.label == "positive")
neg = sum(1 for d in all_deltas if d.label == "negative")
neu = sum(1 for d in all_deltas if d.label == "neutral")
total = len(all_deltas)

print(f"\nLabel distribution across all {total} document-query pairs:")
print(f"  Positive : {pos:3d} ({100*pos/total:.1f}%)")
print(f"  Negative : {neg:3d} ({100*neg/total:.1f}%)")
print(f"  Neutral  : {neu:3d} ({100*neu/total:.1f}%)")

# Delta magnitude distribution
deltas = [d.delta for d in all_deltas]
print(f"\nDelta magnitude statistics:")
print(f"  Min delta : {min(deltas):+.4f}")
print(f"  Max delta : {max(deltas):+.4f}")
print(f"  Mean delta: {sum(deltas)/len(deltas):+.4f}")

# Save full results
save_path = "experiments/logs/gate1_results.jsonl"
from src.training.loss_delta import _save_results
_save_results(results, save_path)
print(f"\nFull results saved to: {save_path}")
print("Review this file manually to confirm label quality.")

# ---------------------------------------------------------------
# Final gate decision
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("GATE 1 FINAL DECISION")
print("=" * 70)
if gate_result["passed"]:
    print("✓ GATE 1 PASSED — Proceed to Week 4 (PubMed corpus build)")
    print(f"  {gate_result['examples_with_positives']}/20 examples have"
          f" positive documents ({100*gate_result['pass_rate']:.0f}%)")
else:
    print("✗ GATE 1 FAILED — Do not proceed to Week 4")
    print("  Review experiments/logs/gate1_results.jsonl")
    print("  Consider lowering tau from 0.05 to 0.02")
    print("  Consider using a different corpus for retrieval")
print("=" * 70)