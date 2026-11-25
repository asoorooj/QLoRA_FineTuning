import torch
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import PeftModel
from accelerate import PartialState

model_name = "Qwen/Qwen2.5-3B-Instruct"

benchmark_prompt = "Evaluate the area under a curve that follows the the curve x * e ^ 2x from points -2 to 5"
config = AutoConfig.from_pretrained(model_name)
config.dtype = torch.float16
config.torch_dtype = torch.float16

bitsAndBytesConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

baseModel = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map={"": PartialState().process_index},
    quantization_config=bitsAndBytesConfig,
    dtype=torch.float16,
    attn_implementation="sdpa",
)


def benchmark_text_generation(model, tokenizer, prompt, runs=10, max_new_tokens=256):
    # Format as a chat conversation if the tokenizer has a chat template
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Appends the assistant start tag
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        # Warmup
        for _ in range(2):
            model.generate(**inputs, max_new_tokens=16, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_tokens = 0
        start = time.perf_counter()

        for _ in range(runs):
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            total_tokens += outputs.shape[1] - input_len

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    # Decode only the assistant's newly generated tokens
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return {
        "ms_per_token": (elapsed / total_tokens) * 1000,
        "tokens_per_sec": total_tokens / elapsed,
        "total_time_s": elapsed,
        "avg_tokens_generated": total_tokens / runs,
        "response": response,
    }


def benchmark_token_prediction(model, tokenizer, prompt, runs=20):
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


def calculate_benchmark_dif(path):

    adapterPath = f"{path}/adapter"
    tokenizer1 = AutoTokenizer.from_pretrained(adapterPath)
    model1 = PeftModel.from_pretrained(baseModel, adapterPath)
    model1.eval()

    before_latency = benchmark_text_generation(model1, tokenizer1, benchmark_prompt, runs=3, max_new_tokens=64)

    mergedPath = f"{path}/merged_model"
    tokenizer2 = AutoTokenizer.from_pretrained(mergedPath)
    model2 = AutoModelForCausalLM.from_pretrained(
        mergedPath,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model2.eval()


    after_latency = benchmark_text_generation(model2, tokenizer2, benchmark_prompt, runs=3, max_new_tokens=64)

    improvement = (
                          (before_latency["ms_per_token"] - after_latency["ms_per_token"]) / before_latency["ms_per_token"]
                  ) * 100

    print(f"before: {before_latency["ms_per_token"]} ms/token, after: {after_latency["ms_per_token"]} ms/token, improvement: {improvement}%")
    return before_latency, after_latency