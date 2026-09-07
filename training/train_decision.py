import json
import fire
import torch
from datasets import load_dataset
from typing import Dict, List, Tuple, Union

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


ALL_LINEAR_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_chat_prompt(dp: Dict[str, str], tokenizer) -> Tuple[str, str]:
    """
    Build Llama3-format chat prompt
    dp["output"] is the serialized discard decision.
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
    Tokenize data point and create labels for loss masking
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


def train(
        base_model: str,
        output_dir: str,

        train_file: str,
        val_file: str = None,

        batch_size: int = 32,
        micro_batch_size: int = 2,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 1024,

        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Union[str, List[str]
                                   ] = json.dumps(ALL_LINEAR_LORA_TARGETS),

        train_on_inputs: bool = False,
):
    print("--- Stage 1: decision LoRA training ---")
    print(f"Base Model: {base_model}")
    print(f"Train File: {train_file}")
    print(f"Val File: {val_file}")
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

    if isinstance(lora_target_modules, str):
        target_modules = json.loads(lora_target_modules)
    else:
        target_modules = lora_target_modules

    print(f"LoRA Target Modules: {target_modules}")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    data_files = {}
    if train_file:
        data_files["train"] = train_file
    else:
        raise ValueError("You must provide --train_file")

    if val_file:
        data_files["validation"] = val_file

    data = load_dataset("json", data_files=data_files)

    train_data = data["train"]
    val_data = data.get("validation")

    print(f"Training data size: {len(train_data)}")
    if val_data:
        print(f"Validation data size: {len(val_data)}")

    train_data = train_data.map(
        lambda dp: tokenize_dp(dp, tokenizer, cutoff_len, train_on_inputs),
        remove_columns=list(train_data.features)
    )
    if val_data:
        val_data = val_data.map(
            lambda dp: tokenize_dp(dp, tokenizer, cutoff_len, train_on_inputs),
            remove_columns=list(val_data.features)
        )

    if batch_size % micro_batch_size != 0:
        raise ValueError(
            "--batch_size must be divisible by --micro_batch_size")
    gradient_accum = batch_size // micro_batch_size

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=num_epochs,
        optim="adamw_torch",
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch" if val_data else "no",
        report_to="none",
        load_best_model_at_end=True if val_data else False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )

    model.config.use_cache = False
    trainer.train()

    print(f"\n--- Stage 1 complete! LoRA adapter saved to: {output_dir} ---")
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
