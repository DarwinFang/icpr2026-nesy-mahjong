import os
import json
import fire
import torch
import sys
from datasets import load_dataset
from typing import Dict, Any, List, Union

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    default_data_collator
)
from peft import (
    LoraConfig,
    PeftModel,
)


def build_chat_prompt(dp: Dict[str, str], tokenizer) -> str:
    """
    (from train_2.py)
    Using Llama 3 chat template
    """
    msgs = [
        {"role": "system", "content": dp["instruction"]},
        {"role": "user", "content": dp["input"]},
    ]
    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    full = prompt + dp["output"] + tokenizer.eos_token
    return prompt, full


def tokenize_dp(dp, tokenizer, cutoff_len, train_on_inputs=False):
    """
    (from train_2.py)
    Tokenize and set -100 labels
    """
    prompt_str, full_str = build_chat_prompt(dp, tokenizer)

    tokenized_full = tokenizer(
        full_str,
        max_length=cutoff_len,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    if not train_on_inputs:
        tokenized_prompt = tokenizer(
            prompt_str,
            max_length=cutoff_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        p_len = len(tokenized_prompt["input_ids"])
        labels = tokenized_full["input_ids"].copy()
        labels[:p_len] = [-100] * p_len
    else:
        labels = tokenized_full["input_ids"].copy()

    tokenized_full["labels"] = labels
    return tokenized_full


def train_v5_pro_fp16(
        base_model: str,
        output_dir: str,

        decision_lora_path: str,

        train_file: str,

        learning_rate: float = 1e-4,
        num_epochs: int = 3,

        batch_size: int = 16,
        micro_batch_size: int = 2,
        cutoff_len: int = 1024,

        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_target_modules: Union[str, List[str]] = '["q_proj", "v_proj"]',

        train_on_inputs: bool = False,
):
    print("--- Stage 2: Explanation LoRA Training (V5-Pro, overlay mode) ---")
    print(f"Base Model: {base_model}")
    print(f"Decision LoRA (frozen): {decision_lora_path}")
    print(f"Train file (V5 format): {train_file}")
    print(f"Output Dir (Explanation LoRA): {output_dir}")

    if not train_file or not decision_lora_path:
        raise ValueError("You must provide --decision_lora_path and --train_file")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("--- Loading base model (BF16, needs significant VRAM) ---")
    device_map = {"": "cuda:0"}
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    print(f"Loading Stage 1 decision LoRA from: {decision_lora_path}")

    model = PeftModel.from_pretrained(
        model,
        decision_lora_path,
        adapter_name="decision_adapter"
    )
    print("--- Stage 1 decision LoRA loaded ---")

    if isinstance(lora_target_modules, str):
        target_modules = json.loads(lora_target_modules)
    else:
        target_modules = lora_target_modules

    print(f"Adding Stage 2 explanation LoRA (explanation_adapter)...")

    expl_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.add_adapter("explanation_adapter", expl_config)


    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "explanation_adapter" in name:
            param.requires_grad = True

    model.set_adapter("explanation_adapter")

    print("--- Stage 2 explanation LoRA added and activated ---")
    model.print_trainable_parameters()

    data = load_dataset("json", data_files={"train": train_file})
    train_data = data["train"]

    print(f"Training data size: {len(train_data)}")

    train_data = train_data.map(
        lambda dp: tokenize_dp(dp, tokenizer, cutoff_len, train_on_inputs),
        remove_columns=list(train_data.features)
    )

    gradient_accum = batch_size // micro_batch_size

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accum,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=None,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    print("--- Starting Stage 2 explanation LoRA training ---")
    trainer.train()

    print(f"\n--- Stage 2 complete! Explanation LoRA saved to: {output_dir} ---")
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train_v5_pro_fp16)
