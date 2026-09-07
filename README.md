# NeSy-Mahjong

Official implementation of **Neuro-Symbolic Instruction Tuning for Explainable Mahjong Agents via Two-Stage Dual-LoRA**, presented at ICPR 2026.

Zhaohao Fang, Junhuai Xu, Jiawei Yu, Hanjie Li, Shuotian Chen, Jiyi Li, and Masaharu Yoshioka

[Paper](https://link.springer.com/chapter/10.1007/978-3-032-31673-8_11) · [Presentation](https://icpr2026orgteam.github.io/OralPres/581.pdf) · [Supplementary material](./Supplementary%20Material.pdf) · [Dataset](https://huggingface.co/datasets/DarwinFang/nesy-mahjong)

## Overview

NeSy-Mahjong is an explainable neuro-symbolic system for *Nanikiru* (what-to-discard) problems in Riichi Mahjong. Its inference pipeline has three steps:

1. **Intuitive decision:** LoRA-A predicts a discard from the observed game state.
2. **Symbolic calculation:** a deterministic calculator derives efficiency, safety, and situational signals, including Shanten, visible Ukeire, Genbutsu, Suji, tile exhaustion, and opponent threat levels.
3. **Grounded explanation:** LoRA-B receives the game state, predicted discard, and symbolic context, then generates an explanation.

The adapters are trained separately so that reasoning training does not overwrite the decision adapter. At inference time, the system switches dynamically from LoRA-A to LoRA-B.

## Results

On the held-out Tenhou benchmark, the four NeSy-Mahjong variants achieve 65.4-67.0% Top-1 decision accuracy and 86.5-88.7% Top-3 accuracy. The strongest commercial baseline reported in the paper, DeepSeek-V3, achieves 32.1% Top-1 accuracy. See the [paper](https://link.springer.com/chapter/10.1007/978-3-032-31673-8_11) for the complete decision, explanation-quality, ablation, and human-alignment results.

## Data

The dataset is derived from 2024 four-player South-round Tenhou Houou logs in which every player is ranked 7-dan or above. Post-Riichi forced discards are excluded. Samples are balanced across offense/defense and early/mid/late contexts, and the train, validation, and test splits are separated chronologically.

| Split | Samples | Dates / source |
|---|---:|---|
| Stage 1 train | 44,034 | Tenhou, 2024-12-20 to 2024-12-29 |
| Stage 1 validation | 4,416 | Tenhou, 2024-12-30 |
| Held-out test | 4,896 | Tenhou, 2024-12-31 |
| Stage 2 train (paper) | 5,544 | Tenhou, 2024-01-01; Gemini-2.0-Flash distillation |
| Mahjong Soul challenge | 20 | Crowdsourced *Nanikiru* problems |

Download the public files from [Hugging Face](https://huggingface.co/datasets/DarwinFang/nesy-mahjong). The currently published `stage2_reasoning.jsonl` is a 2,500-example distilled subset; the experiments in the paper use 5,544 Stage 2 examples.

Each JSONL record contains an instruction, a serialized game state, and an output. Stage 1 outputs use a minimal structured decision:

```json
{
  "instruction": "You are an expert Japanese Mahjong AI...",
  "input": "Game: East 2, 0 Honba...",
  "output": "{\"discard_tile\": \"9p\"}"
}
```

See [`data/README.md`](data/README.md) for the file layout.

## Repository layout

```text
.
├── benchmark/
│   ├── core/                  # Efficiency, safety, and state analysis
│   ├── eval_neurosymbolic.py  # Full LoRA-A -> calculator -> LoRA-B pipeline
│   ├── eval_local.py          # Local baselines and ablations
│   └── eval_api.py            # Commercial API baselines
├── data/                      # Downloaded datasets (not tracked by Git)
├── training/
│   ├── train_decision.py      # Decision-adapter training
│   ├── train_explanation.py   # Frozen LoRA-A + reasoning-adapter training
│   └── train_sequential.py    # Sequential single-LoRA ablation
├── utils/convert_tenhou.py    # Tenhou log conversion
└── Supplementary Material.pdf
```

## Environment

The experiments use Python 3.10 on NVIDIA RTX A6000 GPUs. `requirements.txt` pins the runtime packages for reproducibility.

```bash
conda create -n nesy-mahjong python=3.10 -y
conda activate nesy-mahjong
pip install -r requirements.txt
```

The exact training hyperparameters, prompts, chronological split, and evaluation protocol are documented in the [supplementary material](./Supplementary%20Material.pdf).

## Reproducing the pipeline

The commands below use Qwen2.5-7B-Instruct as an example. Replace the base model with another model evaluated in the paper as needed.

Train the QLoRA decision adapter (LoRA-A):

```bash
python training/train_decision.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_file data/train.jsonl \
  --val_file data/validation.jsonl \
  --output_dir models/qwen-decision
```

Freeze LoRA-A and train the BF16 explanation adapter (LoRA-B):

```bash
python training/train_explanation.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --decision_lora_path models/qwen-decision \
  --train_file data/stage2_reasoning.jsonl \
  --output_dir models/qwen-dual-lora
```

Evaluate the full pipeline with five-beam Top-3 decision decoding and calculator-grounded explanation generation:

```bash
python benchmark/eval_neurosymbolic.py \
  --base_model_path Qwen/Qwen2.5-7B-Instruct \
  --decision_lora_path models/qwen-decision \
  --explanation_lora_path models/qwen-dual-lora/explanation_adapter \
  --split_dir data/split_test \
  --output_dir results/qwen-neurosymbolic
```

Stage 2 saves named PEFT adapters below its output directory. Pass the `explanation_adapter` subdirectory to evaluation. The public repository contains training and evaluation code but does not redistribute base-model or adapter weights.

The sequential single-adapter baseline can be trained separately:

```bash
python training/train_sequential.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --stage1_lora_path models/qwen-decision \
  --train_file data/stage2_reasoning.jsonl \
  --output_dir models/qwen-sequential
```

Use `benchmark/eval_local.py` for local-model baselines and ablations. Its three switches—`use_decision_lora`, `use_explanation_lora`, and `use_knowledge_injection`—control the components independently. `benchmark/eval_api.py` evaluates the commercial baselines; the API key can be passed with `--api_key` or the `MAHJONG_API_KEY` environment variable.

## Citation

The conference is ICPR 2026, while the Springer LNCS volume has the bibliographic publication year 2027. Please use the publisher's year when citing the paper:

```bibtex
@inproceedings{fang2027nesymahjong,
  author    = {Fang, Zhaohao and Xu, Junhuai and Yu, Jiawei and Li, Hanjie and Chen, Shuotian and Li, Jiyi and Yoshioka, Masaharu},
  title     = {Neuro-Symbolic Instruction Tuning for Explainable Mahjong Agents via Two-Stage Dual-LoRA},
  booktitle = {Pattern Recognition},
  series    = {Lecture Notes in Computer Science},
  volume    = {16817},
  pages     = {157--171},
  publisher = {Springer Nature Switzerland},
  year      = {2027},
  doi       = {10.1007/978-3-032-31673-8_11}
}
```
