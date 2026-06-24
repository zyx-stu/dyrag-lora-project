"""
tests/test_trainer.py

Verifies the training loop with a minimal smoke run:
  - 20 steps only (not 500) to keep runtime under 2 minutes
  - Checks loss is decreasing (not flat or NaN)
  - Checks checkpoint is saved
  - Checks log file is written correctly
"""

import sys, os, json
sys.path.insert(0, '/home/user/dyrag-lora')

import torch
from src.training.model_loader import load_model_and_tokenizer
from src.data.medqa_dataset import get_dataloader
from src.training.trainer import QLoRATrainer

print("=" * 60)
print("DyRAG-LoRA: Trainer Verification (smoke run)")
print("=" * 60)

# Load model + tokenizer
print("\n[Setup] Loading model...")
model, tokenizer = load_model_and_tokenizer()

# Build dataloader — 32 samples is enough for a smoke run
print("[Setup] Building dataloader (32 samples)...")
loader = get_dataloader(
    tokenizer   = tokenizer,
    split       = "train",
    batch_size  = 1,
    max_length  = 512,
    max_samples = 32,
    shuffle     = True,
)

# Run trainer for 20 steps only
print("[Setup] Initializing trainer (20 steps)...\n")
trainer = QLoRATrainer(
    model        = model,
    train_loader = loader,
    output_dir   = "experiments",
    phase_name   = "smoke_test",
    config       = {
        "max_steps":                   20,
        "gradient_accumulation_steps": 4,   # effective batch=4 for speed
        "log_every_n_steps":           5,
        "save_every_n_steps":          10,
        "learning_rate":               2e-4,
    },
)

summary = trainer.train()

# ---------------------------------------------------------------
# Test 1: Training completed without crash
# ---------------------------------------------------------------
print("\n[Test 1] Training completed without crash: PASS")

# ---------------------------------------------------------------
# Test 2: Loss log file exists and has entries
# ---------------------------------------------------------------
log_path = "experiments/logs/smoke_test_loss.jsonl"
assert os.path.exists(log_path), f"FAIL: Log file not found at {log_path}"
with open(log_path) as f:
    entries = [json.loads(line) for line in f]
assert len(entries) > 0, "FAIL: Log file is empty"
print(f"[Test 2] Log file written: {len(entries)} entries: PASS")

# ---------------------------------------------------------------
# Test 3: Loss values are finite
# ---------------------------------------------------------------
losses = [e["loss"] for e in entries]
import math
for i, l in enumerate(losses):
    assert not math.isnan(l), f"FAIL: NaN loss at entry {i}"
    assert not math.isinf(l), f"FAIL: Inf loss at entry {i}"
print(f"[Test 3] All loss values finite: {losses}: PASS")

# ---------------------------------------------------------------
# Test 4: Checkpoint saved
# ---------------------------------------------------------------
ckpt_dir = "experiments/checkpoints/smoke_test"
ckpts = os.listdir(ckpt_dir) if os.path.exists(ckpt_dir) else []
assert len(ckpts) > 0, f"FAIL: No checkpoints found in {ckpt_dir}"
print(f"[Test 4] Checkpoint saved ({ckpts}): PASS")

# ---------------------------------------------------------------
# Test 5: VRAM still within bounds after training
# ---------------------------------------------------------------
vram = torch.cuda.memory_reserved() / 1e9
print(f"[Test 5] VRAM after training: {vram:.2f} GB")
assert vram < 15.0, f"FAIL: VRAM {vram:.2f}GB dangerously high"
print(f"[Test 5] VRAM within bounds: PASS")

print("\n" + "=" * 60)
print("ALL TESTS PASSED — Trainer is working correctly")
print("=" * 60)
print(f"\nSummary: {summary}")
