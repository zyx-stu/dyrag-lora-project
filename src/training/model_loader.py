"""
src/training/model_loader.py

Responsible for:
  1. Loading LLaMA-3-8B-Instruct (or any causal LM) in 4-bit NF4 quantization
  2. Attaching LoRA adapters via PEFT
  3. Returning a model ready for QLoRA training

This is the single authoritative model construction function in DyRAG-LoRA.
All other modules import from here — never re-implement model loading inline.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default hyperparameters — locked from the project plan
# ---------------------------------------------------------------------------
# These are not arbitrary: see docs/hyperparameter_rationale.md (Week 3)
# for the full ablation justification. Short version:
#   rank=16     : sweet spot between capacity and parameter efficiency for 8B model
#   alpha=32    : scaling factor alpha/r = 2.0, standard LoRA initialization
#   target_modules: attention projections capture routing decisions (RAG-relevant)
#   dropout=0.05: light regularization, low because QLoRA already acts as regularizer
# ---------------------------------------------------------------------------

DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

DEFAULT_BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
}


def load_model_and_tokenizer(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    lora_config_overrides: dict = None,
    bnb_config_overrides: dict = None,
    device_map: str = "auto",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a causal LM in 4-bit NF4 with LoRA adapters attached.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Default is the main project model.
        Use "microsoft/Phi-3-mini-4k-instruct" for fast debug runs.
    lora_config_overrides : dict, optional
        Override any key in DEFAULT_LORA_CONFIG. E.g., {"r": 8} for rank-8 ablation.
    bnb_config_overrides : dict, optional
        Override any key in DEFAULT_BNB_CONFIG. Rarely needed.
    device_map : str
        "auto" lets accelerate place layers across available devices.
        On a single GPU this always means everything goes to cuda:0.

    Returns
    -------
    model : PeftModel (wraps the quantized base model)
        Ready for training. Only LoRA adapter params have requires_grad=True.
    tokenizer : AutoTokenizer
        Matching tokenizer with pad_token set correctly.
    """

    # ------------------------------------------------------------------
    # 1. Build configs (merge defaults with any overrides)
    # ------------------------------------------------------------------
    bnb_cfg = {**DEFAULT_BNB_CONFIG, **(bnb_config_overrides or {})}
    lora_cfg = {**DEFAULT_LORA_CONFIG, **(lora_config_overrides or {})}

    bnb_config = BitsAndBytesConfig(**bnb_cfg)

    logger.info(f"Loading base model: {model_id}")
    logger.info(f"LoRA config: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, "
                f"targets={lora_cfg['target_modules']}")

    # ------------------------------------------------------------------
    # 2. Load quantized base model
    # ------------------------------------------------------------------
    # torch_dtype=torch.bfloat16 sets the compute dtype for non-quantized
    # operations (layer norms, embeddings). Must match bnb_4bit_compute_dtype.
    # ------------------------------------------------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=torch.bfloat16,
    )

    # ------------------------------------------------------------------
    # 3. Prepare for k-bit training
    # ------------------------------------------------------------------
    # This does two critical things that are easy to forget:
    #   a) Casts layer norm weights to float32 (stability during backprop)
    #   b) Sets requires_grad=False on all base model parameters
    # Without this step, gradients into the quantized weights cause NaN loss.
    # ------------------------------------------------------------------
    base_model = prepare_model_for_kbit_training(base_model)

    # ------------------------------------------------------------------
    # 4. Attach LoRA adapters
    # ------------------------------------------------------------------
    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(base_model, lora_config)

    # ------------------------------------------------------------------
    # 5. Load tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # LLaMA-3 has no pad token by default — set it to eos_token.
    # This is safe for causal LM training (we mask pad positions in loss).
    # Without this, batch collation will fail when sequences have different lengths.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # 6. Log parameter counts — important for the paper's setup section
    # ------------------------------------------------------------------
    _log_trainable_parameters(model)

    return model, tokenizer


def _log_trainable_parameters(model: torch.nn.Module) -> None:
    """
    Print the trainable vs total parameter count.
    This number goes directly into the paper's Experimental Setup section.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    print(f"\nParameter summary:")
    print(f"  Trainable parameters : {trainable:>12,}  ({pct:.4f}%)")
    print(f"  Frozen parameters    : {total - trainable:>12,}")
    print(f"  Total parameters     : {total:>12,}\n")
