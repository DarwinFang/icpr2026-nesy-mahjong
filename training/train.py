import os
import json
import fire
import torch
from datasets import load_dataset
from typing import Dict, Any, List, Union

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
)


def build_chat_prompt(dp: Dict[str, str], tokenizer) -> str:
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


def train_stage2_sequential(
        base_model: str,
        stage1_lora_path: str,
        output_dir: str,
        train_file: str,
        val_file: str = None,

        batch_size: int = 16,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        cutoff_len: int = 1536,

        train_on_inputs: bool = False,
):
    print("--- Sequential Fine-Tuning (Stage 1→2) ---")
    print(f"Base Model: {base_model}")
    print(f"Resuming from Stage 1 LoRA: {stage1_lora_path}")
    print(f"Train File: {train_file}")
    print(f"Output Dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    device_map = {"": "cuda:0"}
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    model = prepare_model_for_kbit_training(model)

    print(f"Loading Stage 1 weights and setting trainable mode...")
    model = PeftModel.from_pretrained(
        model,
        stage1_lora_path,
        is_trainable=True,
        adapter_name="default"
    )

    print("--- Stage 1 LoRA loaded successfully ---")
    model.print_trainable_parameters()

    data_files = {"train": train_file}
    if val_file: data_files["validation"] = val_file

    data = load_dataset("json", data_files=data_files)
    train_data = data["train"]

    print(f"Training data size: {len(train_data)}")

    train_data = train_data.map(
        lambda dp: tokenize_dp(dp, tokenizer, cutoff_len, train_on_inputs),
        remove_columns=list(train_data.features)
    )

    val_dataset = None
    if val_file:
        val_dataset = data["validation"].map(
            lambda dp: tokenize_dp(dp, tokenizer, cutoff_len, train_on_inputs),
            remove_columns=list(data["validation"].features)
        )

    gradient_accum = batch_size // micro_batch_size

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accum,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )

    model.config.use_cache = False

    print("--- Starting sequential fine-tuning ---")
    trainer.train()

    print(f"\n--- Sequential training complete! Merged LoRA saved to: {output_dir} ---")
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train_stage2_sequential)
