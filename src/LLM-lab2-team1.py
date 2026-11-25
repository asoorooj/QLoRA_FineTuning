"""
Lab 2 - Team 1: Parameter-Efficient Fine-Tuning via 4-Bit QLoRA
Graduate Laboratory Series: Deep Learning and Agentic Systems

Objective:
    Master memory-constrained adaptation of foundation models (Qwen-2.5-3B-Instruct
    or Llama-3.2-3B-Instruct) using 4-bit NormalFloat (NF4) quantization, double
    quantization (DQ), and Low-Rank Adaptation (LoRA). Perform systematic empirical
    ablations across low-rank capacities r in {4, 16, 64} and compute adapter merging.

Components to implement:
    1. NF4 Quantization Configuration (BitsAndBytesConfig)
    2. LoRA Adapter Injection via PEFT & Trainable Parameter Verification (<=1.0%)
    3. Peak VRAM & Memory Footprint Profiling Harness
    4. Rank Capacity Ablation & Training Arguments Configuration
    5. Lossless Parameter Merging into Base Weights (W_merged = W_0 + (alpha/r)*B*A)
"""

import os
import gc
from typing import Dict, Any, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset


# ==============================================================================
# Part 1: NF4 Quantization & BitsAndBytes Configuration
# ==============================================================================

def build_bnb_4bit_config() -> BitsAndBytesConfig:
    """
    Configure 4-bit NormalFloat (NF4) quantization with double quantization
    and bfloat16/float16 compute activations.
    
    Returns:
        BitsAndBytesConfig configured for memory-optimized training.
    """
    # --------------------------------------------------------------------------
    # TODO 1.1: Return a BitsAndBytesConfig with:
    #   - load_in_4bit=True
    #   - bnb_4bit_quant_type="nf4"
    #   - bnb_4bit_use_double_quant=True
    #   - bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 1.1: Implement build_bnb_4bit_config.")


# ==============================================================================
# Part 2: LoRA Config & Parameter Ratio Audit
# ==============================================================================

def configure_lora(rank: int, lora_alpha: int = 16, lora_dropout: float = 0.05) -> LoraConfig:
    """
    Construct a LoRA configuration targeting attention projections.
    
    Args:
        rank: Intrinsic rank subspace dimension r (e.g., 4, 16, 64).
        lora_alpha: Scaling parameter alpha.
        lora_dropout: Dropout probability for LoRA layers.
        
    Returns:
        LoraConfig targeting query, key, value, and output projection layers.
    """
    # --------------------------------------------------------------------------
    # TODO 2.1: Construct and return a LoraConfig targeting:
    #   - target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    #   - task_type="CAUSAL_LM"
    #   - bias="none"
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 2.1: Implement configure_lora.")


def audit_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int, float]:
    """
    Calculate the total parameters, trainable parameters, and the trainable ratio.
    
    Returns:
        Tuple of (trainable_params, total_params, percentage).
    """
    trainable_params = 0
    all_params = 0
    # --------------------------------------------------------------------------
    # TODO 2.2: Iterate through model.named_parameters(), accumulate numel(),
    # and compute the percentage: (trainable_params / all_params) * 100.
    # Assert or verify that trainable percentage <= 1.0% to meet parameter ceiling.
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 2.2: Implement audit_trainable_parameters.")


# ==============================================================================
# Part 3: Peak VRAM & Memory Profiling Harness
# ==============================================================================

def get_vram_usage_gb() -> Dict[str, float]:
    """
    Query CUDA memory consumption to ensure execution is within Google Colab
    T4 GPU limits (15 GB ceiling).
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}

    # --------------------------------------------------------------------------
    # TODO 3.1: Use torch.cuda memory tracking functions:
    #   - torch.cuda.memory_allocated() / (1024 ** 3)
    #   - torch.cuda.memory_reserved() / (1024 ** 3)
    #   - torch.cuda.max_memory_allocated() / (1024 ** 3)
    # Return as a dictionary of floats rounded to 3 decimal places.
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 3.1: Implement get_vram_usage_gb.")


# ==============================================================================
# Part 4: Training Setup & Rank Capacity Ablation
# ==============================================================================

def setup_training_args(output_dir: str, learning_rate: float = 2e-4) -> TrainingArguments:
    """
    Setup memory-efficient TrainingArguments for Google Colab Free Tier.
    """
    # --------------------------------------------------------------------------
    # TODO 4.1: Return TrainingArguments configured for zero-budget constraints:
    #   - per_device_train_batch_size=2
    #   - gradient_accumulation_steps=4 (effective batch size 8)
    #   - warmup_ratio=0.03
    #   - max_steps=100 (or num_train_epochs=1)
    #   - logging_steps=10
    #   - optim="paged_adamw_8bit" (to eliminate optimizer memory spikes)
    #   - fp16=not torch.cuda.is_bf16_supported()
    #   - bf16=torch.cuda.is_bf16_supported()
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 4.1: Implement setup_training_args.")


# ==============================================================================
# Part 5: Lossless Parameter Merging for Zero-Latency Export
# ==============================================================================

def merge_and_export_adapter(
    base_model_id: str,
    adapter_checkpoint_dir: str,
    merged_output_dir: str
) -> None:
    """
    Merge low-rank matrices back into base weights in full precision:
        W_merged = W_0 + (alpha / r) * (B @ A)
    Exports a standalone, non-quantized model with zero inference latency overhead.
    """
    print(f"Loading 16-bit unquantized base model {base_model_id} for merging...")
    # --------------------------------------------------------------------------
    # TODO 5.1:
    # 1. Load the base model in FP16 / BF16 (torch_dtype=torch.float16, device_map="cpu" or "auto").
    # 2. Attach the trained adapter via PeftModel.from_pretrained(base_model, adapter_checkpoint_dir).
    # 3. Call peft_model.merge_and_unload() to fuse weights in-place.
    # 4. Save merged model and tokenizer to merged_output_dir.
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 5.1: Implement merge_and_export_adapter.")


# ==============================================================================
# Verification / Empirical Test Harness
# ==============================================================================

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # or "meta-llama/Llama-3.2-3B-Instruct"
    print(f"=== Initializing Lab 2 - Team 1: QLoRA Rank Ablation on {MODEL_ID} ===")

    # Step 1: NF4 Configuration
    try:
        bnb_config = build_bnb_4bit_config()
        print("BitsAndBytes 4-bit config successfully generated.")
    except NotImplementedError as e:
        print(f"Skipping: {e}")
        bnb_config = None

    # Step 2: Test Parameter Cardinality across Ranks r in {4, 16, 64}
    for test_r in [4, 16, 64]:
        try:
            lora_cfg = configure_lora(rank=test_r, lora_alpha=16)
            print(f"LoRA config created for rank r={test_r}.")
        except NotImplementedError as e:
            print(f"Skipping rank {test_r}: {e}")
            break