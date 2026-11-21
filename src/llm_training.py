from accelerate import PartialState
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator
)
import math
import bitsandbytes as bnb

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt

import pandas as pd
import time

from vram_tracker import VRamTracker

# Hyper Parameters
model_name = "Qwen/Qwen2.5-3B-Instruct"
L = 1024 # Sequence length
B = 2 # batch size per-device
G = 8 # gradient accumulation steps
optimizer_name = "paged_adamw_8bit"
ni = 2e-4 # learning rate
# dataset_name = "tatsu-lab/alpaca"
# dataset_name = "nvidia/Nemotron-Math-v2"
dataset_name = "project-numina/NuminaMath-CoT"
benchmark_prompt = "What is the integral of x?"


def train_model(
        model,
        train_dataset,
        eval_dataset,
        vram_tracker,
        epochs=1,
        logging_steps=50,
        eval_steps=50
):

    # Map your original optimizer names to
    # Hugging Face optimizer identifiers.

    OPTIMIZER_MAP = {
        "adamw": "adamw_torch",
        "adamw_8bit": "adamw_bnb_8bit",
        "paged_adamw_8bit": "paged_adamw_8bit",
        "paged_adamw_32bit": "paged_adamw_32bit",
    }

    optimizer = OPTIMIZER_MAP[optimizer_name]

    # --------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------

    # 3 Tri-Rank Empirical Ablation (Hyperparameters defined above)
    training_args = TrainingArguments(

        # Training configuration
        num_train_epochs=epochs,
        per_device_train_batch_size=B,
        per_device_eval_batch_size=B,
        gradient_accumulation_steps=G,

        # Optimizer configuration
        optim=optimizer,
        learning_rate=ni,
        lr_scheduler_type="linear",
        # warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,

        # Mixed precision
        fp16=True,
        bf16=False,

        # Logging
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,

        # Checkpointing
        save_strategy="no",

        # Performance
        dataloader_pin_memory=True,

        # Metrics
        # include_tokens_per_second=True,

        # Miscellaneous
        report_to="none",
        remove_unused_columns=False,

    )

    # --------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------

    # callbacks = [
    #     VRAMTrackingCallback(vram_tracker)
    # ]

    # --------------------------------------------------
    # TRAINER
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        data_collator=default_data_collator,

        # callbacks=callbacks,

    )

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    start_time = time.perf_counter()

    train_result = trainer.train()

    # --------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------

    final_eval = trainer.evaluate()

    runtime = time.perf_counter() - start_time

    # --------------------------------------------------
    # HISTORY EXTRACTION
    # --------------------------------------------------

    history = {
        "training_steps": [],
        "training_losses": [],
        "validation_steps": [],
        "validation_losses": []
    }

    for entry in trainer.state.log_history:

        if "loss" in entry:

            history["training_steps"].append(
                entry["step"]
            )

            history["training_losses"].append(
                entry["loss"]
            )

        if "eval_loss" in entry:

            history["validation_steps"].append(
                entry["step"]
            )

            history["validation_losses"].append(
                entry["eval_loss"]
            )

    # --------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------

    metrics = train_result.metrics

    throughput = metrics.get(
        "train_tokens_per_second",
        0.0
    )

    global_step = trainer.state.global_step

    print("\nTraining Complete")
    print("-------------------------")
    print(f"Optimizer Steps: {global_step}")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Tokens/sec: {throughput:.2f}")
    print(f"Final Validation Loss: {final_eval['eval_loss']:.4f}")

    return {
        "history": history,
        "global_step": global_step,
        "runtime": runtime,
        "tokens_per_second": throughput,
        "final_eval": final_eval,
        "trainer": trainer
    }

def benchmark(model, tokenizer, prompt, runs=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    for _ in range(3):
        model.generate(**inputs, max_new_tokens=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(runs):
        model.generate(**inputs, max_new_tokens=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return (elapsed / runs) * 1000

def run_training_experiment(rank, alpha, output_directory):

    vramTracker = VRamTracker()

    vramTracker.resetVRamMemoryStat()

    # Force config to float16 so SFTTrainer/TRL never sees bfloat16
    config = AutoConfig.from_pretrained(model_name)
    config.dtype = torch.float16
    config.torch_dtype = torch.float16

    bitsAndBytesConfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 1 Quantized Model Instantiation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map={"": PartialState().process_index},
        quantization_config=bitsAndBytesConfig,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )

    # 4 VRAM profiling
    vramTracker.recordPeakVRam("Base Loading")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


    # 4 VRAM profiling
    vramTracker.recordPeakVRam("Tokenizer Loading")

    # 2 Target Subspace Adaptation
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # 4 VRAM profiling
    vramTracker.recordPeakVRam("Adapter Injection")

    model.print_trainable_parameters()

    dataset = Dataset.from_list(list(load_dataset(
        dataset_name,
        split="train",
        streaming=True
    ).take(10_000)))

    def tokenize_function(examples):

        texts = [
            problem + "\n" + solution + tokenizer.eos_token
            for problem, solution in zip(
                examples["problem"],
                examples["solution"]
            )
        ]

        return tokenizer(
            texts,
            truncation=False,
            add_special_tokens=True
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    def group_texts(examples):

        concatenated = {
            key: sum(examples[key], [])
            for key in examples.keys()
        }

        total_length = len(concatenated["input_ids"])

        total_length = (total_length // L) * L

        result = {
            key: [
                values[i:i + L]
                for i in range(0, total_length, L)
            ]
            for key, values in concatenated.items()
        }

        result["labels"] = [
            ids.copy()
            for ids in result["input_ids"]
        ]

        return result

    packed_dataset = tokenized_dataset.map(
        group_texts,
        batched=True
    )
    dataset = packed_dataset.train_test_split(
        test_size=0.1,
        seed=42
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]




    # Training loop

    results = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        vram_tracker=vramTracker,
        epochs=1,
        logging_steps=100,
        eval_steps=100
    )

    history = results["history"]

    print("Runtime:", results["runtime"])
    print("Tokens/sec:", results["tokens_per_second"])
    print("Steps:", results["global_step"])

    # print("VRAM:", vramTracker.vram_usage)

    # 4 VRAM profiling
    vramTracker.recordPeakVRam("Adapter Injection")

    # 5 Adapter Merging & Zero-Overhead Inference

    model.eval()

    before_latency = benchmark(
        model,
        tokenizer,
        benchmark_prompt,
        runs=10
    )

    print(f"Before merging: {before_latency:.4f} ms/token")

    # Save LoRA adapter
    adapter_directory = f"{output_directory}/adapter"

    model.save_pretrained(adapter_directory)
    tokenizer.save_pretrained(adapter_directory)

    # Load trained LoRA adapter

    # Merge adapter weights into base model
    merged_model = model.merge_and_unload(
        safe_merge=True
    )

    merged_model.eval()

    merged_directory = f"{output_directory}/merged_model"

    merged_model.save_pretrained(merged_directory)
    tokenizer.save_pretrained(merged_directory)

    after_latency = benchmark(
        merged_model,
        tokenizer,
        benchmark_prompt,
        runs=10
    )

    print(f"After merging: {after_latency:.4f} ms/token")
    improvement = (
                          (before_latency - after_latency) / before_latency
                  ) * 100

    print(f"Latency improvement: {improvement:.2f}%")

    # print(vramTracker.vram_usage)

    plt.figure(figsize=(10, 6))

    print(history)

    plt.plot(
        history["training_steps"],
        history["training_losses"],
        label="Training Loss"
    )

    plt.plot(
        history["validation_steps"],
        history["validation_losses"],
        label="Validation Loss"
    )

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss (Rank={rank}, Alpha={alpha})")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{output_directory}/loss_curve.png")
    plt.show()

    vram_df = pd.DataFrame([
        {
            "phase": label,
            "peak_vram_gb": value
        }
        for label, value in vramTracker.vram_usage.items()
    ])

    vram_df.to_csv(
        f"{output_directory}/vram_usage.csv",
        index=False
    )

    # 4 VRAM profiling
    maxVramUsage = vramTracker.recordPeakVRam("Total Max VRAM Usage")
    print(f"Max VRAM Usage: {maxVramUsage:.2f} GB")

    del model, merged_model, tokenizer, dataset, train_dataset, eval_dataset, vramTracker, tokenized_dataset, packed_dataset
    torch.cuda.empty_cache()
