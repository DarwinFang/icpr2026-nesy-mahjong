import json
import sys
import torch
import argparse
import os
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Set

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
    from analyzer import FullGameStateAnalyzer

    print("FullGameStateAnalyzer imported successfully")
except ImportError:
    print("\nFATAL: mahjong_full_analysis.py not found.")
    sys.exit(1)

VALID_TILES_list = [
    "1m", "2m", "3m", "4m", "5m", "0m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "0p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "0s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "Wh", "G", "R"
]
VALID_TILES: Set[str] = set(VALID_TILES_list)


def format_comparative_data(analysis_result, target_tile):
    """
    Convert hand analysis into a structured decision comparison table.
    target_tile: target_tile is the Stage 1 model prediction
    """
    if not analysis_result or 'tile_analysis' not in analysis_result:
        return "(objective data computation failed)"

    analysis_dict = analysis_result['tile_analysis']
    tenpai_info = analysis_result.get('tenpai_estimates', {})
    process_info = analysis_result.get('process_estimates', "unknown turn")

    tenpai_str_list = [f"P{pid}: {status}" for pid, status in tenpai_info.items()]
    tenpai_str = ", ".join(tenpai_str_list)

    global_context = f"【{process_info}】Tenpai probability: {tenpai_str}"

    valid_tiles = []
    target_data = None

    for tile, data in analysis_dict.items():
        if 'error' in data: continue

        item = {
            'tile': tile,
            'shanten': data.get('shanten', 99),
            'ukeire': data.get('ukeire', 0),
            'safety': data.get('safety_analysis', '-') or "-"
        }
        valid_tiles.append(item)

        if tile == target_tile:
            target_data = item

    if not target_data:
        target_data = {'tile': target_tile, 'shanten': 99, 'ukeire': 0, 'safety': 'unknown'}
        valid_tiles.append(target_data)

    valid_tiles.sort(key=lambda x: (x['shanten'], -x['ukeire']))

    display_list = valid_tiles[:4]
    if target_data not in display_list:
        display_list.append(target_data)

    lines = []
    lines.append(f"Global context: {global_context}")
    lines.append("[Discard option comparison] (sorted by efficiency):")

    for item in display_list:
        t = item['tile']
        s = item['shanten']
        u = item['ukeire']
        safe = item['safety']

        marker = " 👈(actual decision)" if t == target_tile else ""

        lines.append(f"- [discard {t}]: {s} shanten,  accepts {u} tiles | {safe}{marker}")

    return "\n".join(lines)


class MahjongNeuroSymbolicModel:
    def __init__(self,
                 base_model_path: str,
                 decision_lora_path: Optional[str] = None,
                 explanation_lora_path: Optional[str] = None):

        print(f"--- Loading base model (BF16): {base_model_path} ---")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        self.has_decision_adapter = False
        self.has_explanation_adapter = False

        if decision_lora_path:
            print(f"--- Loading decision LoRA: {decision_lora_path} ---")
            model = PeftModel.from_pretrained(
                model, decision_lora_path, adapter_name="decision_adapter"
            )
            self.has_decision_adapter = True

        if explanation_lora_path:
            print(f"--- Loading explanation LoRA: {explanation_lora_path} ---")
            if self.has_decision_adapter:
                model.load_adapter(explanation_lora_path, adapter_name="explanation_adapter")
            else:
                model = PeftModel.from_pretrained(
                    model, explanation_lora_path, adapter_name="explanation_adapter"
                )
            self.has_explanation_adapter = True

        model.eval()
        self.model = model

        if isinstance(self.tokenizer.eos_token_id, list):
            self.stop_token_id = self.tokenizer.eos_token_id[0]
        else:
            self.stop_token_id = self.tokenizer.eos_token_id

        self.json_pattern = re.compile(r':\s*"([^"]+)"')
        self.start_pattern = re.compile(r'^([1-9][mps]|0[mps]|[ESWN]|Wh|G|R)')
        self.thoughts_pattern = re.compile(r'"thoughts":\s*"([^"]+)"', re.DOTALL)

        print("--- Initializing mahjong calculator (Python) ---")
        self.analyzer = FullGameStateAnalyzer()
        print("--- Analyzer initialized ---")

    def _build_prompt(self, instruction: str, user_input: str) -> str:
        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": user_input}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def predict(self, raw_input_text: str) -> Tuple[List[str], str, str]:
        """
        Run inference (benchmark, robust extraction).
        Returns: (Top3 decisions, explanation, Stage2 prompt preview)
        """

        top3_discards = []

        DECISION_INSTRUCTION = "You are an expert Japanese Mahjong AI. Analyze the following game state and determine which tile to discard."
        decision_prompt = self._build_prompt(DECISION_INSTRUCTION, raw_input_text)

        if self.has_decision_adapter:
            self.model.set_adapter("decision_adapter")

        inputs = self.tokenizer(decision_prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=64, num_beams=5, num_return_sequences=3,
                    eos_token_id=self.stop_token_id, pad_token_id=self.tokenizer.pad_token_id, early_stopping=True
                )

            raw_candidates = []
            input_len = inputs.input_ids.shape[1]

            for output_ids in outputs:
                decoded_text = self.tokenizer.decode(output_ids[input_len:], skip_special_tokens=True).strip()

                tile = None

                if decoded_text in VALID_TILES:
                    tile = decoded_text

                elif self.json_pattern.search(decoded_text):
                    tile = self.json_pattern.search(decoded_text).group(1)

                else:
                    candidates = re.findall(r'([1-9][mps]|0[mps]|[ESWN]|Wh|G|R)', decoded_text)
                    if candidates: tile = candidates[0]

                if tile and tile in VALID_TILES:
                    raw_candidates.append(tile)

            seen = set()
            for x in raw_candidates:
                if x not in seen: top3_discards.append(x); seen.add(x)
            while len(top3_discards) < 3: top3_discards.append("INVALID")

        except Exception as e:
            print(f"Stage 1 Error: {e}")
            top3_discards = ["Error", "Error", "Error"]

        best_discard = top3_discards[0]
        model_thoughts = "N/A"
        formatted_calc_data = ""

        if best_discard not in ["INVALID", "Error"]:
            try:
                symbolic_result = self.analyzer.analyze(raw_input_text)
                formatted_calc_data = format_comparative_data(symbolic_result, best_discard)
            except Exception as e:
                formatted_calc_data = f"（Calculator error: {e}）"

        explanation_input_final = ""

        if self.has_explanation_adapter and best_discard not in ["INVALID", "Error"]:
            self.model.set_adapter("explanation_adapter")

            explanation_input_final = (
                f"【Game state】:\n{raw_input_text}\n\n"
                f"【Objective data】\n{formatted_calc_data}\n\n"
                f"【Decision】:\ndiscard  {best_discard}"
            )

            EXPLANATION_INSTRUCTION = "You are a rigorous mahjong analyst. Based on the game and calculation data (including discard comparison), analyze the decision."

            explanation_prompt = self._build_prompt(EXPLANATION_INSTRUCTION, explanation_input_final)
            explanation_inputs = self.tokenizer(explanation_prompt, return_tensors="pt").to(self.model.device)

            try:
                with torch.no_grad():
                    explanation_outputs = self.model.generate(
                        **explanation_inputs,
                        max_new_tokens=512,
                        do_sample=True, temperature=0.3, top_p=0.9,
                        eos_token_id=self.stop_token_id, pad_token_id=self.tokenizer.pad_token_id
                    )

                exp_input_len = explanation_inputs.input_ids.shape[1]
                generated_text = self.tokenizer.decode(explanation_outputs[0][exp_input_len:],
                                                       skip_special_tokens=True).strip()

                thoughts_match = self.thoughts_pattern.search(generated_text)
                if thoughts_match:
                    model_thoughts = thoughts_match.group(1)
                else:
                    model_thoughts = generated_text

            except Exception as e:
                model_thoughts = f"Stage 2 Error: {e}"

            return top3_discards, model_thoughts, explanation_input_final

        return top3_discards, "Skipped Stage 2", ""


def load_test_samples(filename: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filename):
        return []
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except:
                pass
    return samples


def save_results(results_data: List[Dict[str, Any]], filename: str):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run Mahjong Final V5 Pro Benchmark")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--decision_lora_path", type=str, default=None)
    parser.add_argument("--explanation_lora_path", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default="split_test")
    parser.add_argument("--output_dir", type=str, default="benchmark_results_v5_pro")
    args = parser.parse_args()

    TARGET_BUCKETS = [
        "test_early_offense.jsonl", "test_mid_offense.jsonl", "test_late_offense.jsonl",
        "test_early_defense.jsonl", "test_mid_defense.jsonl", "test_late_defense.jsonl"
    ]

    try:
        model = MahjongNeuroSymbolicModel(
            args.base_model_path,
            args.decision_lora_path,
            args.explanation_lora_path
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    summary_stats = []

    print("\n" + "=" * 60)
    print(f"🚀 Starting V5-Pro full benchmark")
    print(f"🧠 Pipeline: Stage 1 -> Calculator -> Stage 2")
    print("=" * 60 + "\n")

    for bucket_file in TARGET_BUCKETS:
        file_path = os.path.join(args.split_dir, bucket_file)
        test_samples = load_test_samples(file_path)

        if not test_samples:
            print(f"⏩ Skipping {bucket_file}")
            continue

        bucket_results = []
        correct_top1 = 0
        total_count = 0

        print(f"📂 Evaluating bucket: {bucket_file} (samples: {len(test_samples)})")

        for sample in tqdm(test_samples, desc=f"Testing {bucket_file}"):
            raw_input = sample.get("input")
            ground_truth = sample.get("output")

            if not raw_input or not ground_truth: continue

            try:
                top3_discards, model_thoughts, prompt_used = model.predict(raw_input)

                best_decision = top3_discards[0]
                is_correct = (best_decision == ground_truth)
                if is_correct: correct_top1 += 1
                total_count += 1

                bucket_results.append({
                    "raw_input": raw_input,
                    "ground_truth": ground_truth,
                    "model_decision": best_decision,
                    "model_thoughts": model_thoughts,
                    "stage2_prompt_preview": prompt_used[:500] + "...",
                    "is_correct": is_correct
                })

            except Exception as e:
                print(f"Sample Error: {e}")

        acc = (correct_top1 / total_count * 100) if total_count > 0 else 0

        result_filename = f"result_{bucket_file.replace('.jsonl', '.json')}"
        save_path = os.path.join(args.output_dir, result_filename)
        save_results(bucket_results, save_path)

        summary_stats.append({"bucket": bucket_file, "total": total_count, "acc": acc})
        print(f"   ✅ Done. Acc: {acc:.1f}%")

    print("\n" + "=" * 40)
    for stat in summary_stats:
        print(f"{stat['bucket']:<25} | {stat['acc']:>6.2f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
