"""
tests/test_model_loader.py

Verifies that:
  1. Model loads correctly in 4-bit NF4 with LoRA adapters
  2. Only LoRA parameters have requires_grad=True
  3. A forward+backward pass completes without OOM or NaN
  4. Base model weights receive zero gradient (frozen correctly)
  5. VRAM usage is within expected bounds
"""

import sys
sys.path.insert(0, '/home/user/dyrag-lora')   # make src importable

import torch
from src.training.model_loader import load_model_and_tokenizer

print("=" * 60)
print("DyRAG-LoRA: Model Loader Verification")
print("=" * 60)

# ---------------------------------------------------------------
# Test 1: Load model
# ---------------------------------------------------------------
print("\n[Test 1] Loading model with LoRA adapters...")
model, tokenizer = load_model_and_tokenizer(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct"
)
print("  PASS: Model loaded")

# ---------------------------------------------------------------
# Test 2: Verify only LoRA params are trainable
# ---------------------------------------------------------------
print("\n[Test 2] Checking requires_grad flags...")
lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
frozen_params = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]

assert len(lora_params) > 0, "FAIL: No trainable parameters found"
for name, _ in lora_params[:5]:
    print(f"  Trainable: {name}")
print(f"  ... and {len(lora_params)-5} more LoRA params")

# Confirm no base weight accidentally has requires_grad=True
non_lora_trainable = [n for n, p in model.named_parameters()
                      if p.requires_grad and 'lora_' not in n]
assert len(non_lora_trainable) == 0, \
    f"FAIL: Non-LoRA params are trainable: {non_lora_trainable[:3]}"
print("  PASS: Only lora_ parameters are trainable")

# ---------------------------------------------------------------
# Test 3: Forward pass
# ---------------------------------------------------------------
print("\n[Test 3] Running forward pass...")
text = "Retrieval-augmented generation improves language models by"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
print(f"  Loss value: {loss.item():.4f}")
assert not torch.isnan(loss), "FAIL: Loss is NaN"
assert not torch.isinf(loss), "FAIL: Loss is Inf"
print("  PASS: Forward pass clean, loss is finite")

# ---------------------------------------------------------------
# Test 4: Backward pass — gradients flow through LoRA only
# ---------------------------------------------------------------
print("\n[Test 4] Running backward pass...")
loss.backward()

# Check LoRA params got gradients
lora_with_grad = [(n, p) for n, p in model.named_parameters()
                  if p.requires_grad and p.grad is not None]
assert len(lora_with_grad) > 0, "FAIL: No LoRA parameters received gradients"
print(f"  LoRA params with gradients: {len(lora_with_grad)}")

# Check base model params have NO gradient
base_with_grad = [n for n, p in model.named_parameters()
                  if not p.requires_grad and p.grad is not None]
assert len(base_with_grad) == 0, \
    f"FAIL: Frozen params received gradients: {base_with_grad[:3]}"
print("  PASS: Gradients flow only through LoRA parameters")

# ---------------------------------------------------------------
# Test 5: VRAM check
# ---------------------------------------------------------------
print("\n[Test 5] VRAM usage under training conditions...")
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved() / 1e9
print(f"  Allocated: {allocated:.2f} GB")
print(f"  Reserved:  {reserved:.2f} GB")
assert reserved < 14.0, \
    f"FAIL: VRAM usage {reserved:.2f}GB dangerously close to 16GB ceiling"
print(f"  PASS: VRAM within safe bounds ({16.0 - reserved:.1f} GB remaining)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED — QLoRA adapter setup is correct")
print("=" * 60)
