# NeSy-Mahjong (ICPR 2026)

**Neuro-Symbolic Instruction Tuning for Explainable Mahjong Agents via Two-Stage Dual-LoRA**

[Zhaohao Fang](https://github.com/), Junhuai Xu, Jiawei Yu, Hanjie Li, Shuotian Chen, Jiyi Li, Masaharu Yoshioka  

---

## Overview

NeSy-Mahjong is an interpretable neuro-symbolic system for **Nanikiru (what-to-discard)** problems in Riichi Mahjong. It uses a decoupled two-stage dual-LoRA architecture:

1. **Stage 1 — Intuitive Decision**: A decision adapter (LoRA-A) is aligned with professional-level discard intuition from high-level Tenhou game logs.
2. **Stage 2 — Grounded Reasoning**: A separate reasoning adapter (LoRA-B) is trained via teacher-student distillation, with a **symbolic calculator** injecting objective Mahjong metrics (Shanten, Ukeire, Genbutsu) into the generation context.

This separation prevents catastrophic forgetting and ensures explanations are mathematically grounded.

**Key results**: Our sub-10B open-source models significantly outperform SOTA commercial models (DeepSeek-V3, Qwen-Plus) in both decision accuracy and explanation quality.

---

## Repository Structure

```
icpr2026-nesy-mahjong/
├── README.md                          # This file
├── Supplementary Material.pdf         # Paper supplementary material
│
├── data/                              # Datasets (symlinks to full data)
│   ├── train.jsonl                    # Stage 1 training: game state → discard decision
│   ├── test.jsonl                     # Test set
│   ├── validation.jsonl               # Validation set
│   ├── stage2_reasoning.jsonl  # Stage 2 synthetic data (2,500 samples)
│
├── training/                          # Training scripts
│   ├── stage1_align.py              # Stage 1: decision alignment (LoRA-A)
│   ├── stage2_reason.py             # Stage 2: reasoning distillation (LoRA-B)
│   └── train.py                     # Sequential: stage1 → stage2
│
├── benchmark/                         # Evaluation scripts
│   ├── eval_neurosymbolic.py         # Full neuro-symbolic eval
│   ├── eval_vanilla.py               # Baseline eval
│   ├── eval_knowledge.py             # Knowledge-injection variant
│   ├── eval_api.py                   # API benchmark (Qwen, DeepSeek, Gemini)
│   │
│   └── benchmark_utils/               # Shared utility modules
│       ├── engine.py                    # Shanten, Ukeire, Yaku calculator
│       ├── parser.py                    # Game state parser
│       ├── analyzer.py                  # Full state analysis
│       └── defense.py                   # Genbutsu, Suji, safety
│
├── models/                            # LoRA adapter weights (download separately)
│   ├── yi-lora-stage-1-decision/      # Yi-6B / Yi-9B Stage 1 adapter
│   ├── yi-lora-stage-2-explanation/   # Yi Stage 2 adapter
│   ├── yi-lora-sequential-final/      # Yi sequential training final
│   ├── llama3-lora-stage-1-decision/  # LLaMA3-8B Stage 1 adapter
│   ├── llama3-lora-stage-2-explanation/# LLaMA3 Stage 2 adapter
│   ├── llama3-lora-sequential-final/  # LLaMA3 sequential training final
│   ├── qwen-lora-stage-1-decision/    # Qwen2.5-7B Stage 1 adapter
│   ├── qwen-lora-stage-2-explanation/ # Qwen Stage 2 adapter
│   ├── qwen-lora-sequential-final/    # Qwen sequential training final
│   ├── deepseek-lora-stage-1-decision/# DeepSeek Stage 1 adapter
│   ├── deepseek-lora-stage-2-explanation/ # DeepSeek Stage 2 adapter
│   └── deepseek-lora-sequential-final/# DeepSeek sequential training final
│
└── results/                           # Experiment output (not tracked in git)
```

---

## Datasets

### Data Format

Each line in the `.jsonl` files follows this structure:

```json
{
  "instruction": "You are an expert Japanese Mahjong AI...",
  "input": "Game: 东2局, 0 Honba...\nHand: 2m 2m 3p...\nDrawn Tile: 9p\n...",
  "output": "{\"discard_tile\": \"9p\"}"
}
```

- **Stage 1 data** (`train.jsonl`, `test.jsonl`, `validation.jsonl`): Derived from high-level Tenhou game logs. Output is the discard decision.
- **Stage 2 data** (`synthetic_neuro_symbolic_*.jsonl`): Teacher-distilled reasoning data with symbolic calculator injection.

### Data Split

| Split | Samples | Source |
|-------|---------|--------|
| train.jsonl | ~5,000 | Tenhou Houou games |
| validation.jsonl | ~500 | Tenhou Houou games |
| test.jsonl | ~500 | Tenhou Houou games |
| stage2_reasoning.jsonl | 2,500 | Teacher distillation |

---

## Training

### Stage 1: Decision Alignment

Fine-tunes a LoRA adapter (LoRA-A) to predict the correct discard tile.

```bash
python training/stage1_align.py \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --data_path "data/train.jsonl" \
  --output_dir "models/qwen-lora-stage-1-decision" \
  --lora_r 16 --lora_alpha 32
```

### Stage 2: Explanation Distillation

Freezes LoRA-A, then trains LoRA-B to generate coherent explanations grounded by symbolic metrics.

```bash
python training/stage2_reason.py \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --decision_adapter "models/qwen-lora-stage-1-decision" \
  --data_path "data/stage2_reasoning.jsonl" \
  --output_dir "models/qwen-lora-stage-2-explanation" \
  --lora_r 16 --lora_alpha 32
```

### Sequential Training

Runs stage 1 → stage 2 in a single script:

```bash
python training/train.py \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --stage1_data "data/train.jsonl" \
  --stage2_data "data/stage2_reasoning.jsonl" \
  --output_dir "models/qwen-lora-sequential-final"
```

---

## Evaluation

### Neuro-Symbolic Benchmark (with knowledge injection)

```bash
python benchmark/eval_neurosymbolic.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --decision_adapter "models/qwen-lora-stage-1-decision" \
  --reasoning_adapter "models/qwen-lora-stage-2-explanation" \
  --test_file "data/test.jsonl" \
  --use_knowledge_injection
```

### Flexible Benchmark (without knowledge injection)

```bash
python benchmark/eval_vanilla.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --decision_adapter "models/qwen-lora-stage-1-decision" \
  --test_file "data/test.jsonl"
```

### API Benchmark (for commercial models)

```bash
python benchmark/eval_api.py \
  --model "qwen" \
  --test_file "data/test.jsonl" \
  --api_key $DASHSCOPE_API_KEY
```

Supported API models: `qwen` (Qwen-Plus), `deepseek` (DeepSeek-V3), `gemini` (Gemini-2.0-Flash), `glm`, `kimi`.

---

## Models

| Base Model | Size | Decision Adapter | Reasoning Adapter |
|-----------|------|-----------------|-------------------|
| Qwen2.5-7B-Instruct | 7B | ✓ | ✓ |
| Yi-6B-Chat | 6B | ✓ | ✓ |
| LLaMA 3.1 8B | 8B | ✓ | ✓ |
| DeepSeek-LLM 7B | 7B | ✓ | ✓ |

The adapters are standard PEFT/LoRA modules (~10-50 MB each). Load with:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "models/qwen-lora-stage-1-decision")
```

---

## Environment

```bash
conda create -n nesy python=3.10 -y && conda activate nesy
pip install -r requirements.txt
```

Verified: Python 3.11, torch 2.9.1, CUDA 12.2.

---

## Model Weights

**Do NOT redistribute** — download from HuggingFace. Some models require license agreements (e.g. LLaMA needs Meta approval).

| Model | HuggingFace ID | Size |
|-------|---------------|------|
| Qwen2.5-7B | Qwen/Qwen2.5-7B-Instruct | ~15 GB |
| LLaMA 3.1 8B | meta-llama/Llama-3.1-8B-Instruct | ~16 GB (Meta approval) |
| Yi-6B | 01-ai/Yi-6B-Chat | ~12 GB |
| DeepSeek 7B | deepseek-ai/deepseek-llm-7b-chat | ~14 GB |


### API Baselines (for comparison)

| Model | Provider | API ID |
|-------|----------|--------|
| Qwen-Plus | DashScope | qwen-plus |
| DeepSeek-V3 | DashScope | deepseek-v3 |
| Gemini 2.0 Flash | Google | gemini-2.0-flash |
| GLM-4.6 | Zhipu | glm-4.6 |
| Kimi K2 | Moonshot | Moonshot-Kimi-K2-Instruct |

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir vanilla_models/Qwen2.5-7B-Instruct
```

## Citation

```bibtex
@inproceedings{fang2026nesy,
  title={Neuro-Symbolic Instruction Tuning for Explainable Mahjong Agents via Two-Stage Dual-LoRA},
  author={Fang, Zhaohao and Xu, Junhuai and Yu, Jiawei and Li, Hanjie and Chen, Shuotian and Li, Jiyi and Yoshioka, Masaharu},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2026}
}
```

---

## License

MIT License (to be confirmed).
