import argparse
import os
import time
from tqdm import tqdm
from typing import List, Tuple

from openai import OpenAI

if __package__:
    from .common import (
        TEST_BUCKETS,
        VALID_TILES,
        extract_generated_discard,
        extract_thoughts,
        format_comparative_data,
        get_ground_truth_discard,
        load_jsonl,
        parse_bool,
        save_json,
    )
    from .core.analyzer import FullGameStateAnalyzer
else:
    from common import (
        TEST_BUCKETS,
        VALID_TILES,
        extract_generated_discard,
        extract_thoughts,
        format_comparative_data,
        get_ground_truth_discard,
        load_jsonl,
        parse_bool,
        save_json,
    )
    from core.analyzer import FullGameStateAnalyzer

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

        print(
            f"--- Initializing API: {self.model_id} | URL: {self.base_url} ---")

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

        self.use_knowledge_injection = use_knowledge_injection

        self.analyzer = None
        if use_knowledge_injection:
            self.analyzer = FullGameStateAnalyzer()
            print("--- Analyzer initialized ---")

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

        decision_output = self._call_api(
            DECISION_INSTRUCTION, raw_input_text, temperature=0.0)

        top3_discards = []
        tile = extract_generated_discard(decision_output, prefer_dollar=True)

        if tile and tile in VALID_TILES:
            top3_discards.append(tile)
        else:
            top3_discards.append("INVALID")

        while len(top3_discards) < 3:
            top3_discards.append("INVALID")

        best_discard = top3_discards[0]
        model_thoughts = "Skipped Stage 2"
        explanation_input_final = ""

        if not run_explanation or best_discard in ["INVALID", "Error"]:
            return top3_discards, model_thoughts, explanation_input_final

        if self.use_knowledge_injection and self.analyzer:
            try:
                symbolic_result = self.analyzer.analyze(raw_input_text)
                formatted_calc_data = format_comparative_data(
                    symbolic_result, best_discard)
            except Exception as e:
                formatted_calc_data = f"（Calculator error: {e}）"

            explanation_input_final = (
                f"【牌局详情】:\n{raw_input_text}\n\n"
                f"【客观数据与对比】\n{formatted_calc_data}\n\n"
                f"【已定决策】:\n切 {best_discard}"
            )
        else:
            explanation_input_final = (
                f"【牌局详情】:\n{raw_input_text}\n\n"
                f"【已定决策】:\n切 {best_discard}"
            )

        EXPLANATION_INSTRUCTION = (
            "你是一个严谨的麻将战术分析师。请基于牌局和客观计算数据（含切牌对比表），分析为什么应该切这张牌。\n"
            "要求：字数不超过150字。"
        )

        explanation_output = self._call_api(
            EXPLANATION_INSTRUCTION, explanation_input_final, temperature=0.3)

        model_thoughts = extract_thoughts(explanation_output)

        return top3_discards, model_thoughts, explanation_input_final


def main():
    parser = argparse.ArgumentParser(
        description="Run API-based Mahjong Benchmark")
    parser.add_argument(
        "--model_name", choices=sorted(MODEL_CONFIG), required=True,
        help="Commercial model configuration to evaluate",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("MAHJONG_API_KEY"),
        help="API key (or set MAHJONG_API_KEY)",
    )

    parser.add_argument("--split_dir", type=str, default="split_test")
    parser.add_argument("--output_dir", type=str,
                        default="benchmark_results_api")

    parser.add_argument("--use_knowledge_injection",
                        type=parse_bool, default=True)
    parser.add_argument("--explanation_limit", type=int, default=40)

    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api_key or set MAHJONG_API_KEY")

    try:
        model = MahjongAPIModel(
            args.model_name, args.api_key, args.use_knowledge_injection)
    except Exception as e:
        print(f"Error: {e}")
        return

    summary_stats = []

    print("\n" + "=" * 60)
    print(f"Starting API evaluation: {args.model_name}")
    print(f"Knowledge injection: {args.use_knowledge_injection}")
    print(
        f"Explanation sample: first {args.explanation_limit} records per bucket")
    print("=" * 60 + "\n")

    for bucket_file in TEST_BUCKETS:
        file_path = os.path.join(args.split_dir, bucket_file)
        test_samples = load_jsonl(file_path)

        if not test_samples:
            print(f"Skipping missing or empty bucket: {bucket_file}")
            continue

        bucket_results = []
        correct_top1 = 0
        total_count = 0

        print(f"Evaluating {bucket_file} ({len(test_samples)} samples)")

        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing")):
            raw_input = sample.get("input")
            ground_truth_output = sample.get("output")
            if not raw_input or ground_truth_output is None:
                continue

            ground_truth = get_ground_truth_discard(ground_truth_output)
            if ground_truth not in VALID_TILES:
                continue

            do_explain = (i < args.explanation_limit)

            try:
                top3_discards, model_thoughts, prompt_used = model.predict(
                    raw_input, run_explanation=do_explain)

                best_decision = top3_discards[0]
                is_correct = (best_decision == ground_truth)
                if is_correct:
                    correct_top1 += 1
                total_count += 1

                bucket_results.append({
                    "raw_input": raw_input,
                    "ground_truth": ground_truth,
                    "model_decision": best_decision,
                    "model_thoughts": model_thoughts,
                    "hit_top1": is_correct,
                })

            except Exception as e:
                print(f"Sample Error: {e}")

        acc1 = (correct_top1 / total_count * 100) if total_count > 0 else 0

        model_suffix = args.model_name
        if not args.use_knowledge_injection:
            model_suffix += "_no_injection"

        result_filename = f"result_{bucket_file.replace('.jsonl', '')}_{model_suffix}.json"
        save_path = os.path.join(args.output_dir, result_filename)
        save_json(bucket_results, save_path)

        summary_stats.append({
            "bucket": bucket_file,
            "total": total_count,
            "acc1": acc1
        })
        print(f"Finished {bucket_file}: Top-1 {acc1:.1f}%")

    print("\n" + "=" * 60)
    print(f"{'Bucket Name':<25} | {'Total':<6} | {'Top-1':<8}")
    print("-" * 60)
    for stat in summary_stats:
        print(
            f"{stat['bucket']:<25} | {stat['total']:<6} | {stat['acc1']:>6.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
