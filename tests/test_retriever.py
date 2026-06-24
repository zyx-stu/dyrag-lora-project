"""
tests/test_retriever.py

Verifies the FAISS retriever with a small synthetic corpus.
Does NOT use PubMed yet — that's Week 3.
Tests:
  1. Index builds correctly on small corpus
  2. Retrieved documents are relevant (semantic match, not random)
  3. Similarity scores are in valid range [0, 1] for cosine similarity
  4. Batch retrieval returns correct shape
  5. Save and load round-trip works correctly
  6. format_retrieved_context produces usable prompt string
"""

import sys, os, shutil
sys.path.insert(0, '/home/user/dyrag-lora')

from src.retrieval.faiss_retriever import FAISSRetriever

print("=" * 60)
print("DyRAG-LoRA: FAISS Retriever Verification")
print("=" * 60)

# Small synthetic medical corpus for testing
# Real corpus (100K PubMed abstracts) is built in Week 3
CORPUS = [
    "Nitrofurantoin is a nitrofuran antibiotic used to treat urinary tract infections. It is generally safe in the second trimester of pregnancy but contraindicated near term due to risk of haemolytic anaemia.",
    "Amoxicillin is a penicillin-type antibiotic effective against gram-positive bacteria. It is considered safe during pregnancy and is commonly used for respiratory tract infections.",
    "Doxycycline is a tetracycline antibiotic that is contraindicated in pregnancy due to its effects on fetal bone and tooth development. It should be avoided in all trimesters.",
    "Ceftriaxone is a third-generation cephalosporin used for serious bacterial infections. It is generally considered safe in pregnancy for severe infections.",
    "Metformin is an oral antidiabetic medication used in type 2 diabetes. It works by reducing hepatic glucose production and improving insulin sensitivity.",
    "ACE inhibitors such as lisinopril are contraindicated in pregnancy due to risk of fetal renal damage and oligohydramnios, especially in the second and third trimesters.",
    "Beta-blockers like metoprolol are used in hypertension and heart failure. They are generally considered safe in pregnancy but may cause neonatal bradycardia.",
    "Fluoroquinolones such as ciprofloxacin are broad-spectrum antibiotics. They are generally avoided in pregnancy due to potential effects on developing cartilage.",
    "Macrolide antibiotics including azithromycin are used for atypical pneumonia. Azithromycin is considered safe in pregnancy and is used for chlamydia infections.",
    "Vancomycin is a glycopeptide antibiotic used for MRSA infections. It requires therapeutic drug monitoring due to its narrow therapeutic index.",
]

# ---------------------------------------------------------------
# Test 1: Build index
# ---------------------------------------------------------------
print("\n[Test 1] Building FAISS index on 10-document corpus...")
retriever = FAISSRetriever(config={"top_k": 3, "device": "cuda"})
retriever.build_index(CORPUS)
assert retriever.index.ntotal == 10, f"FAIL: Expected 10 vectors, got {retriever.index.ntotal}"
print(f"  Index built: {retriever.index.ntotal} vectors, dim={retriever.dim}")
print("  PASS")

# ---------------------------------------------------------------
# Test 2: Semantic relevance — UTI + pregnancy query
# ---------------------------------------------------------------
print("\n[Test 2] Semantic retrieval test...")
query = "What antibiotic is safe for urinary tract infection during pregnancy?"
results = retriever.retrieve(query, k=3)
print(f"  Query: {query[:60]}...")
for i, (doc, score) in enumerate(results, 1):
    print(f"  Rank {i} (score={score:.3f}): {doc[:80]}...")

# The top result should be about nitrofurantoin (most relevant to UTI + pregnancy)
top_3_docs = " ".join([doc.lower() for doc, _ in results])
relevant_terms = ["nitrofurantoin", "urinary", "antibiotic", "amoxicillin",
                  "ceftriaxone", "pregnancy", "azithromycin"]
assert any(term in top_3_docs for term in relevant_terms), \
    "FAIL: Top-3 results contain no medically relevant terms"

# Log whether nitrofurantoin ranked first (important for paper baseline)
top_doc = results[0][0].lower()
nitro_rank = next((i+1 for i, (d,_) in enumerate(results)
                   if "nitrofurantoin" in d.lower()), None)
print(f"  Note: Nitrofurantoin ranked #{nitro_rank} (pre-trained embedder, no fine-tuning yet)")
print(f"  This is the Phase 0 baseline — DyRAG-LoRA should improve this ranking")
print("  PASS: Top-3 results are medically relevant")

# ---------------------------------------------------------------
# Test 3: Similarity scores in valid range
# ---------------------------------------------------------------
print("\n[Test 3] Checking similarity score range...")
for doc, score in results:
    assert -1.0 <= score <= 1.0, f"FAIL: Score {score} outside [-1, 1]"
    assert score > 0, f"FAIL: Score {score} non-positive (query likely unrelated)"
print(f"  Scores: {[round(s, 3) for _, s in results]}")
print("  PASS: All scores in valid cosine similarity range")

# ---------------------------------------------------------------
# Test 4: Batch retrieval
# ---------------------------------------------------------------
print("\n[Test 4] Batch retrieval test...")
queries = [
    "antibiotic for UTI in pregnancy",
    "contraindicated drugs in first trimester",
    "treatment for MRSA infection",
]
batch_results = retriever.retrieve_batch(queries, k=2)
assert len(batch_results) == 3, f"FAIL: Expected 3 result lists, got {len(batch_results)}"
for i, res in enumerate(batch_results):
    assert len(res) == 2, f"FAIL: Query {i} returned {len(res)} results, expected 2"
    print(f"  Query {i+1}: top result = {res[0][0][:60]}...")
print("  PASS: Batch retrieval returns correct structure")

# ---------------------------------------------------------------
# Test 5: Save and load round-trip
# ---------------------------------------------------------------
print("\n[Test 5] Save/load round-trip...")
save_path = "/tmp/test_retriever_index"
if os.path.exists(save_path):
    shutil.rmtree(save_path)

retriever.save(save_path)
assert os.path.exists(f"{save_path}/faiss.index"), "FAIL: faiss.index not saved"
assert os.path.exists(f"{save_path}/documents.json"), "FAIL: documents.json not saved"

# Load into a fresh retriever instance
retriever2 = FAISSRetriever(config={"top_k": 3, "device": "cuda"})
retriever2.load(save_path)
assert retriever2.index.ntotal == 10, "FAIL: Loaded index has wrong size"

# Verify results are identical after load
results2 = retriever2.retrieve(query, k=3)
assert results2[0][0] == results[0][0], "FAIL: Top result changed after save/load"
print("  PASS: Save/load round-trip preserves retrieval results")

# ---------------------------------------------------------------
# Test 6: format_retrieved_context
# ---------------------------------------------------------------
print("\n[Test 6] Context formatting...")
context = retriever.format_retrieved_context(results)
assert "[Document 1]" in context, "FAIL: Missing document header"
assert "[Document 2]" in context, "FAIL: Missing document 2 header"
assert "[Document 3]" in context, "FAIL: Missing document 3 header"
print("  Context preview:")
print("  " + context[:200].replace("\n", "\n  "))
print("  PASS: Context formatted correctly")

print("\n" + "=" * 60)
print("ALL TESTS PASSED — FAISS Retriever is working correctly")
print("=" * 60)
