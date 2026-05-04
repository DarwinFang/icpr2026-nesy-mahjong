import json
import sys
import argparse
import os
import re
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
    from analyzer import FullGameStateAnalyzer

    print("FullGameStateAnalyzer imported successfully")
except ImportError:
    print("Warning: mahjong_full_analysis.py not found.")
    print("This will cause an error if --use_knowledge_injection is True.")

VALID_TILES_list = [
    "1m", "2m", "3m", "4m", "5m", "0m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "0p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "0s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "Wh", "G", "R"
]
VALID_TILES: Set[str] = set(VALID_TILES_list)

MODEL_CONFIG = {
    "qwen": {
        "model_id": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "deepseek": {
        "model_id": "deepseek-v3",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "glm": {
        "model_id": "glm-4.6",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "kimi": {
        "model_id": "Moonshot-Kimi-K2-Instruct",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "gemini": {
        "model_id": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
    }
}


def clean_tile_str(tile_str: str) -> str:
    if not tile_str: return ""
    return tile_str.replace("*", "").strip()


def format_comparative_data(analysis_result, target_tile):
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
        if tile == target_tile: target_data = item

    if not target_data:
        target_data = {'tile': target_tile, 'shanten': 99, 'ukeire': 0, 'safety': 'unknown'}
        valid_tiles.append(target_data)

    valid_tiles.sort(key=lambda x: (x['shanten'], -x['ukeire']))
    display_list = valid_tiles[:4]
    if target_data not in display_list: display_list.append(target_data)

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


class MahjongAPIModel:
    def __init__(self,
                 model_key: str,
                 api_key: str,
                 use_knowledge_injection: bool = True):

        if model_key not in MODEL_CONFIG:
            raise ValueError(f"Model '{model_key}' not found in config.")

        self.config = MODEL_CONFIG[model_key]
        self.model_id = self.config["model_id"]
        self.base_url = self.config["base_url"]

        print(f"--- Initializing API: {self.model_id} | URL: {self.base_url} ---")

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

        self.use_knowledge_injection = use_knowledge_injection

        self.json_pattern = re.compile(r':\s*"([^"]+)"')
        self.dollar_pattern = re.compile(r'\$\s*([1-9][mps]|0[mps]|[ESWN]|Wh|G|R)')
        self.thoughts_pattern = re.compile(r'"thoughts":\s*"([^"]+)"', re.DOTALL)

        self.analyzer = None
        if use_knowledge_injection:
            try:
                self.analyzer = FullGameStateAnalyzer()
                print("--- Analyzer initialized ---")
            except Exception as e:
                print(f"❌ Analyzer init failed: {e}")

    def _call_api(self, system_prompt: str, user_input: str, temperature: float = 0.0) -> str:
        """Unified API call interface"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=temperature,
                    top_p=0.9,
                    max_tokens=512
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                print(f"API Error: {e}")
                return ""
        return ""

    def predict(self, raw_input_text: str, run_explanation: bool = False) -> Tuple[List[str], str, str]:

        DECISION_INSTRUCTION = (
            "You are an expert Japanese Mahjong AI. Analyze the following game state and determine which tile to discard. "
            "**No more than 150 words.** "
            "At the end of your analysis, please output your final decision on a new line starting with the symbol '$', "
            "like this:\n$ 4s"
        )

        decision_output = self._call_api(DECISION_INSTRUCTION, raw_input_text, temperature=0.0)

        top3_discards = []
        tile = None
        clean_text = decision_output.strip()

        dollar_match = self.dollar_pattern.search(clean_text)
        if dollar_match:
            tile = dollar_match.group(1)
        elif self.json_pattern.search(clean_text):
            tile = self.json_pattern.search(clean_text).group(1)
        else:
            candidates = re.findall(r'([1-9][mps]|0[mps]|[ESWN]|Wh|G|R)', clean_text)
            if candidates: tile = candidates[0]

        if tile and tile in VALID_TILES:
            top3_discards.append(tile)
        else:
            top3_discards.append("INVALID")

        while len(top3_discards) < 3: top3_discards.append("INVALID")

        best_discard = top3_discards[0]
        model_thoughts = "Skipped Stage 2"
        explanation_input_final = ""

        if not run_explanation or best_discard in ["INVALID", "Error"]:
            return top3_discards, model_thoughts, explanation_input_final

        if self.use_knowledge_injection and self.analyzer:
            try:
                symbolic_result = self.analyzer.analyze(raw_input_text)
                formatted_calc_data = format_comparative_data(symbolic_result, best_discard)
            except Exception as e:
                formatted_calc_data = f"（Calculator error: {e}）"

            explanation_input_final = (
                f"【Game state】:\n{raw_input_text}\n\n"
                f"【Objective data】\n{formatted_calc_data}\n\n"
                f"【Decision】:\ndiscard  {best_discard}"
            )
        else:
            explanation_input_final = (
                f"【Game state】:\n{raw_input_text}\n\n"
                f"【Decision】:\ndiscard  {best_discard}"
            )

        EXPLANATION_INSTRUCTION = (
            "You are a rigorous mahjong analyst. Based on the game state and objective calculation data (including the discard comparison table), analyze why this tile should be discarded.\n"
            "Limit: 150 characters max."
        )

        explanation_output = self._call_api(EXPLANATION_INSTRUCTION, explanation_input_final, temperature=0.3)

        thoughts_match = self.thoughts_pattern.search(explanation_output)
        if thoughts_match:
            model_thoughts = thoughts_match.group(1)
        else:
            model_thoughts = explanation_output

        return top3_discards, model_thoughts, explanation_input_final


def load_test_samples(filename: str, num_samples: int) -> List[Dict[str, Any]]:
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return []
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples: break
            try:
                samples.append(json.loads(line))
            except:
                pass
    return samples


def save_results(results_data: List[Dict[str, Any]], filename: str):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)


def get_ground_truth_discard(output_field: Any) -> str:
    if isinstance(output_field, str):
        try:
            return clean_tile_str(json.loads(output_field).get("discard_tile", output_field.strip()))
        except:
            return clean_tile_str(output_field)
    if isinstance(output_field, dict): return clean_tile_str(output_field.get("discard_tile", "N/A"))
    return "N/A"


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="Run API-based Mahjong Benchmark")
    parser.add_argument("--model_name", type=str, required=True, help="Model key: qwen, gemini, gpt-4o")
    parser.add_argument("--api_key", type=str, required=True, help="API Key")

    parser.add_argument("--split_dir", type=str, default="split_test")
    parser.add_argument("--output_dir", type=str, default="benchmark_results_api")

    parser.add_argument("--use_knowledge_injection", type=str2bool, default=True)
    parser.add_argument("--explanation_limit", type=int, default=40)

    args = parser.parse_args()

    try:
        model = MahjongAPIModel(args.model_name, args.api_key, args.use_knowledge_injection)
    except Exception as e:
        print(f"Error: {e}")
        return

    summary_stats = []

    TARGET_BUCKETS = [
        "majsoul_challenge_20.jsonl"
    ]

    print("\n" + "=" * 60)
    print(f"🚀 Starting API Benchmark: {args.model_name}")
    print(f"🔧 Config: Knowledge Injection={args.use_knowledge_injection}")
    print(f"📝 Explanation sampling limit: first {args.explanation_limit}  per bucket")
    print("=" * 60 + "\n")

    for bucket_file in TARGET_BUCKETS:
        file_path = os.path.join(args.split_dir, bucket_file)
        test_samples = load_test_samples(file_path, 10000)

        if not test_samples:
            print(f"⏩ Skipping {bucket_file}")
            continue

        bucket_results = []
        correct_top1 = 0
        total_count = 0

        print(f"📂 Evaluating bucket: {bucket_file} (total: {len(test_samples)})")

        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing")):
            raw_input = sample.get("input")
            ground_truth_output = sample.get("output")
            ground_truth_output = sample.get("output", "1m")
            if not raw_input: continue

            ground_truth = get_ground_truth_discard(ground_truth_output)
            if not ground_truth or ground_truth == "N/A":
                ground_truth = "1m"

            do_explain = (i < args.explanation_limit)

            try:
                top3_discards, model_thoughts, prompt_used = model.predict(raw_input, run_explanation=do_explain)

                best_decision = top3_discards[0]
                is_correct = (best_decision == ground_truth)
                if is_correct: correct_top1 += 1
                total_count += 1

                hit_top2 = is_correct
                hit_top3 = is_correct

                bucket_results.append({
                    "raw_input": raw_input if do_explain else "",
                    "ground_truth": ground_truth,
                    "model_decision": best_decision,
                    "model_top3": top3_discards,
                    "model_thoughts": model_thoughts,
                    "hit_top1": is_correct,
                    "hit_top2": hit_top2,
                    "hit_top3": hit_top3,
                })

            except Exception as e:
                print(f"Sample Error: {e}")

        acc1 = (correct_top1 / total_count * 100) if total_count > 0 else 0

        model_suffix = args.model_name
        if not args.use_knowledge_injection:
            model_suffix += "_no_injection"

        result_filename = f"result_{bucket_file.replace('.jsonl', '')}_{model_suffix}.json"
        save_path = os.path.join(args.output_dir, result_filename)
        save_results(bucket_results, save_path)

        summary_stats.append({
            "bucket": bucket_file,
            "total": total_count,
            "acc1": acc1
        })
        print(f"   ✅ Done. Acc Top1: {acc1:.1f}%")

    print("\n" + "=" * 60)
    print(f"{'Bucket Name':<25} | {'Total':<6} | {'Top-1':<8}")
    print("-" * 60)
    for stat in summary_stats:
        print(f"{stat['bucket']:<25} | {stat['total']:<6} | {stat['acc1']:>6.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
