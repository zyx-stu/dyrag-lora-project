"""
src/data/medqa_dataset.py

Responsible for:
  1. Loading MedQA-USMLE from HuggingFace
  2. Formatting each example into LLaMA-3-Instruct chat template
  3. Tokenizing with correct loss masking (-100 on prompt tokens)
  4. Returning a DataLoader ready for the training loop

Design decision: prompt tokens get label=-100 (ignored by cross-entropy).
The model is supervised only on the assistant's answer tokens.
This is standard practice for instruction fine-tuning — see Alpaca, Vicuna.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical expert answering USMLE-style questions. "
    "Read the question carefully, consider all options, and provide the correct answer."
)


def format_options(options: Dict[str, str]) -> str:
    """
    Convert options dict {"A": "Ampicillin", ...} to formatted string.

    Example output:
        A: Ampicillin
        B: Ceftriaxone
        C: Doxycycline
        D: Nitrofurantoin
    """
    return "\n".join(f"{k}: {v}" for k, v in options.items())


def format_answer(answer: str, answer_idx: str) -> str:
    """
    Format the target answer string.

    We include both the index and the full answer text because:
    - The loss is computed over all answer tokens, not just "D"
    - Full text teaches the model to associate option text with correctness
    - Matches evaluation format (exact match on answer_idx)

    Example: "The answer is D: Nitrofurantoin"
    """
    return f"The answer is {answer_idx}: {answer}"


def build_chat_prompt(
    tokenizer: AutoTokenizer,
    question: str,
    options: Dict[str, str],
    answer: Optional[str] = None,
    answer_idx: Optional[str] = None,
) -> str:
    """
    Build the full LLaMA-3-Instruct formatted prompt.

    For training (answer provided): includes assistant response + eot_id
    For inference (no answer): stops at the generation prompt marker

    LLaMA-3 special tokens used:
      <|begin_of_text|>          — start of sequence
      <|start_header_id|>        — role header open
      <|end_header_id|>          — role header close
      <|eot_id|>                 — end of turn

    We use tokenizer.apply_chat_template() to handle these correctly
    rather than manually concatenating strings — this is safer because
    the template is guaranteed to match what the model was trained on.
    """
    user_content = (
        f"Question: {question}\n\n"
        f"Options:\n{format_options(options)}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    if answer is not None and answer_idx is not None:
        # Training mode: include the answer as the assistant turn
        messages.append({
            "role": "assistant",
            "content": format_answer(answer, answer_idx)
        })
        # add_generation_prompt=False because we're providing the full response
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # Inference mode: stop at generation prompt, let model complete
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MedQADataset(Dataset):
    """
    PyTorch Dataset for MedQA-USMLE fine-tuning.

    Each item returns:
        input_ids      : torch.LongTensor  [seq_len]
        attention_mask : torch.LongTensor  [seq_len]
        labels         : torch.LongTensor  [seq_len]  — prompt tokens = -100

    The -100 masking means CrossEntropyLoss ignores prompt positions.
    Only the assistant answer tokens contribute to the training loss.
    """

    IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss default ignore index

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        split: str = "train",
        max_length: int = 512,
        max_samples: Optional[int] = None,
        dataset_id: str = "GBaker/MedQA-USMLE-4-options",
    ):
        """
        Parameters
        ----------
        tokenizer   : must be the LLaMA-3 tokenizer with pad_token set
        split       : "train" or "test"
        max_length  : maximum sequence length in tokens (512 is safe for 16GB VRAM)
        max_samples : if set, truncate dataset (useful for debug runs)
        dataset_id  : HuggingFace dataset path
        """
        self.tokenizer  = tokenizer
        self.max_length = max_length

        logger.info(f"Loading MedQA dataset: {dataset_id}, split={split}")
        raw = load_dataset(dataset_id, split=split)

        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))
            logger.info(f"Truncated to {len(raw)} samples for debug run")

        self.data = raw
        logger.info(f"Dataset ready: {len(self.data)} examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data[idx]

        # ------------------------------------------------------------------
        # Step 1: Build the full formatted prompt (prompt + answer)
        # ------------------------------------------------------------------
        full_text = build_chat_prompt(
            tokenizer  = self.tokenizer,
            question   = row["question"],
            options    = row["options"],
            answer     = row["answer"],
            answer_idx = row["answer_idx"],
        )

        # ------------------------------------------------------------------
        # Step 2: Build the prompt-only text (no answer)
        # We tokenize both to find exactly where the answer tokens start.
        # This boundary is used for loss masking.
        # ------------------------------------------------------------------
        prompt_only = build_chat_prompt(
            tokenizer  = self.tokenizer,
            question   = row["question"],
            options    = row["options"],
            answer     = None,
            answer_idx = None,
        )

        # ------------------------------------------------------------------
        # Step 3: Tokenize full text
        # ------------------------------------------------------------------
        tokenized = self.tokenizer(
            full_text,
            max_length     = self.max_length,
            truncation     = True,
            padding        = False,   # padding happens in collate_fn
            return_tensors = "pt",
        )
        input_ids      = tokenized["input_ids"].squeeze(0)       # [seq_len]
        attention_mask = tokenized["attention_mask"].squeeze(0)  # [seq_len]

        # ------------------------------------------------------------------
        # Step 4: Find prompt boundary for loss masking
        # Tokenize prompt only (no padding, no truncation) to get its length.
        # All tokens before this boundary get label=-100.
        # ------------------------------------------------------------------
        prompt_len = len(self.tokenizer(
            prompt_only,
            add_special_tokens=False,
        )["input_ids"])

        # ------------------------------------------------------------------
        # Step 5: Build labels — copy input_ids, mask prompt tokens
        # ------------------------------------------------------------------
        labels = input_ids.clone()
        # Mask everything up to and including the prompt
        # (prompt_len is the index of the first answer token)
        labels[:prompt_len] = self.IGNORE_INDEX

        # If the sequence was truncated and no answer tokens remain,
        # skip this example by masking everything.
        # This prevents training on incomplete answers.
        if (labels != self.IGNORE_INDEX).sum() == 0:
            labels[:] = self.IGNORE_INDEX
            logger.debug(f"Example {idx} fully masked (truncated answer)")

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Pad a batch of variable-length sequences to the same length.

        Padding strategy:
          input_ids      → pad with tokenizer.pad_token_id (right-pad)
          attention_mask → pad with 0 (don't attend to padding)
          labels         → pad with -100 (don't compute loss on padding)

        Right-padding is used (not left-padding) because causal LMs
        process tokens left-to-right — left-padding would shift the
        meaningful content away from position 0 which can confuse the
        position embeddings.
        """
        input_ids      = [item["input_ids"]      for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels         = [item["labels"]         for item in batch]

        # Find the longest sequence in this batch
        max_len = max(ids.size(0) for ids in input_ids)

        padded_input_ids, padded_masks, padded_labels = [], [], []

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            pad_len = max_len - ids.size(0)

            padded_input_ids.append(
                torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_masks.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids":      torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_masks),
            "labels":         torch.stack(padded_labels),
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def get_dataloader(
    tokenizer:   AutoTokenizer,
    split:       str  = "train",
    batch_size:  int  = 1,
    max_length:  int  = 512,
    max_samples: Optional[int] = None,
    shuffle:     bool = True,
    num_workers: int  = 0,
) -> DataLoader:
    """
    Build and return a DataLoader for the specified split.

    num_workers=0 is intentional for training — HuggingFace tokenizers
    use forked processes internally and can deadlock with num_workers>0
    unless spawn is configured explicitly. Keep at 0 unless you profile
    and confirm a bottleneck here.
    """
    dataset = MedQADataset(
        tokenizer   = tokenizer,
        split       = split,
        max_length  = max_length,
        max_samples = max_samples,
    )

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = MedQADataset.collate_fn,
        num_workers = num_workers,
        pin_memory  = True,   # faster CPU→GPU transfer
    )
