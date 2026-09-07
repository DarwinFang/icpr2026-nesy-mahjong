import torch
import argparse
import os
from tqdm import tqdm
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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


class LocalMahjongModel:
    def __init__(self,
                 base_model_path: str,
                 decision_lora_path: Optional[str] = None,
                 explanation_lora_path: Optional[str] = None,
                 use_decision_lora: bool = True,
                 use_explanation_lora: bool = True,
                 use_knowledge_injection: bool = True):

        print(f"--- Loading base model (BF16): {base_model_path} ---")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        self.use_decision_lora = use_decision_lora
        self.use_explanation_lora = use_explanation_lora
        self.use_knowledge_injection = use_knowledge_injection

        self.has_decision_adapter = False
        self.has_explanation_adapter = False

        if decision_lora_path and use_decision_lora:
            print(f"--- Loading decision LoRA: {decision_lora_path} ---")
            model = PeftModel.from_pretrained(
                model, decision_lora_path, adapter_name="decision_adapter"
            )
            self.has_decision_adapter = True
        elif not use_decision_lora:
            print("Decision LoRA disabled; using the base model")

        if explanation_lora_path and use_explanation_lora:
            print(f"--- Loading explanation LoRA: {explanation_lora_path} ---")
            if self.has_decision_adapter:
                model.load_adapter(explanation_lora_path,
                                   adapter_name="explanation_adapter")
            else:
                model = PeftModel.from_pretrained(
                    model, explanation_lora_path, adapter_name="explanation_adapter"
                )
            self.has_explanation_adapter = True
        elif not use_explanation_lora:
            print("Explanation LoRA disabled")

        model.eval()
        self.model = model

        if isinstance(self.tokenizer.eos_token_id, list):
            self.stop_token_id = self.tokenizer.eos_token_id[0]
        else:
            self.stop_token_id = self.tokenizer.eos_token_id

        self.analyzer = None
        if use_knowledge_injection:
            self.analyzer = FullGameStateAnalyzer()
            print("--- Analyzer initialized ---")

    def _build_prompt(self, instruction: str, user_input: str) -> str:
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": user_input}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def predict(self, raw_input_text: str, run_explanation: bool = False) -> Tuple[List[str], str]:
        """
        Returns: (Top3 decisions, explanation)
        """

        top3_discards = []

        if self.has_decision_adapter:
            DECISION_INSTRUCTION = "You are an expert Japanese Mahjong AI. Analyze the following game state and determine which tile to discard."
        else:
            DECISION_INSTRUCTION = (
                "You are an expert Japanese Mahjong AI. Analyze the following game state and determine which tile to discard. "
                "No more than 150 words. At the end, output the final decision on a new line beginning with '$', "
                "for example: $ 4s"
            )
        decision_prompt = self._build_prompt(
            DECISION_INSTRUCTION, raw_input_text)

        if self.has_decision_adapter:
            self.model.set_adapter("decision_adapter")

        inputs = self.tokenizer(
            decision_prompt, return_tensors="pt").to(self.model.device)

        try:
            context_manager = self.model.disable_adapter() if (
                not self.has_decision_adapter and self.has_explanation_adapter) else torch.no_grad()

            with context_manager:
                with torch.no_grad():
                    num_beams = 5 if self.has_decision_adapter else 1
                    num_returns = 3 if self.has_decision_adapter else 1
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64 if self.has_decision_adapter else 512,
                        num_beams=num_beams,
                        num_return_sequences=num_returns,
                        eos_token_id=self.stop_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        early_stopping=True
                    )

            raw_candidates = []
            input_len = inputs.input_ids.shape[1]
            for output_ids in outputs:
                decoded_text = self.tokenizer.decode(
                    output_ids[input_len:], skip_special_tokens=True).strip()

                tile = extract_generated_discard(
                    decoded_text, prefer_dollar=not self.has_decision_adapter
                )

                if tile and tile in VALID_TILES:
                    raw_candidates.append(tile)

            seen = set()
            for x in raw_candidates:
                if x not in seen:
                    top3_discards.append(x)
                    seen.add(x)
            while len(top3_discards) < 3:
                top3_discards.append("INVALID")

        except Exception as e:
            print(f"Stage 1 Error: {e}")
            top3_discards = ["Error", "Error", "Error"]

        best_discard = top3_discards[0]
        model_thoughts = "Skipped Stage 2"

        if not run_explanation or best_discard in ["INVALID", "Error"]:
            return top3_discards, model_thoughts

        explanation_input_final = ""

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

        EXPLANATION_INSTRUCTION = "你是一个严谨的麻将战术分析师。请基于牌局和客观计算数据（含切牌对比表），分析为什么应该切这张牌。"
        explanation_prompt = self._build_prompt(
            EXPLANATION_INSTRUCTION, explanation_input_final)
        explanation_inputs = self.tokenizer(
            explanation_prompt, return_tensors="pt").to(self.model.device)

        def run_generate():
            return self.model.generate(
                **explanation_inputs,
                max_new_tokens=512,
                do_sample=True, temperature=0.3, top_p=0.9,
                eos_token_id=self.stop_token_id, pad_token_id=self.tokenizer.pad_token_id
            )

        try:
            with torch.no_grad():
                if self.has_explanation_adapter:
                    self.model.set_adapter("explanation_adapter")
                    explanation_outputs = run_generate()
                elif self.has_decision_adapter:
                    with self.model.disable_adapter():
                        explanation_outputs = run_generate()
                else:
                    explanation_outputs = run_generate()

            exp_input_len = explanation_inputs.input_ids.shape[1]
            generated_text = self.tokenizer.decode(explanation_outputs[0][exp_input_len:],
                                                   skip_special_tokens=True).strip()

            model_thoughts = extract_thoughts(generated_text)

        except Exception as e:
            model_thoughts = f"Stage 2 Error: {e}"

        return top3_discards, model_thoughts


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local models and ablations")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--decision_lora_path", type=str, default=None)
    parser.add_argument("--explanation_lora_path", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default="split_test")
    parser.add_argument("--output_dir", type=str,
                        default="benchmark_results_local")

    parser.add_argument("--use_decision_lora", type=parse_bool, default=True)
    parser.add_argument("--use_explanation_lora",
                        type=parse_bool, default=True)
    parser.add_argument("--use_knowledge_injection",
                        type=parse_bool, default=True)

    parser.add_argument("--explanation_limit", type=int, default=40)

    args = parser.parse_args()

    try:
        model = LocalMahjongModel(
            args.base_model_path,
            args.decision_lora_path,
            args.explanation_lora_path,
            args.use_decision_lora,
            args.use_explanation_lora,
            args.use_knowledge_injection
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    summary_stats = []

    print("\n" + "=" * 60)
    print("Starting local-model evaluation")
    print(
        f"Configuration: decision LoRA={args.use_decision_lora}, "
        f"explanation LoRA={args.use_explanation_lora}, "
        f"knowledge injection={args.use_knowledge_injection}"
    )
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
        correct_top2 = 0
        correct_top3 = 0
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
                top3_discards, model_thoughts = model.predict(
                    raw_input, run_explanation=do_explain)

                is_hit_top1 = (ground_truth == top3_discards[0])
                is_hit_top2 = (ground_truth in top3_discards[:2])
                is_hit_top3 = (ground_truth in top3_discards)

                if is_hit_top1:
                    correct_top1 += 1
                if is_hit_top2:
                    correct_top2 += 1
                if is_hit_top3:
                    correct_top3 += 1
                total_count += 1

                bucket_results.append({
                    "raw_input": raw_input,
                    "ground_truth": ground_truth,
                    "model_top3": top3_discards,
                    "model_thoughts": model_thoughts,
                    "hit_top1": is_hit_top1,
                    "hit_top2": is_hit_top2,
                    "hit_top3": is_hit_top3
                })

            except Exception as e:
                print(f"Sample Error: {e}")

        acc1 = (correct_top1 / total_count * 100) if total_count > 0 else 0
        acc2 = (correct_top2 / total_count * 100) if total_count > 0 else 0
        acc3 = (correct_top3 / total_count * 100) if total_count > 0 else 0

        variant = "_".join([
            "decision-lora" if model.has_decision_adapter else "base-decision",
            "explanation-lora" if model.has_explanation_adapter else "base-explanation",
            "knowledge" if model.use_knowledge_injection else "no-knowledge",
        ])
        result_filename = f"result_{bucket_file.replace('.jsonl', '')}_{variant}.json"
        save_path = os.path.join(args.output_dir, result_filename)
        save_json(bucket_results, save_path)

        summary_stats.append({
            "bucket": bucket_file,
            "total": total_count,
            "acc1": acc1,
            "acc2": acc2,
            "acc3": acc3
        })
        print(f"Finished {bucket_file}: Top-1 {acc1:.1f}% | Top-3 {acc3:.1f}%")

    print("\n" + "=" * 60)
    print(f"{'Bucket Name':<25} | {'Total':<6} | {'Top-1':<8} | {'Top-2':<8} | {'Top-3':<8}")
    print("-" * 60)
    for stat in summary_stats:
        print(
            f"{stat['bucket']:<25} | {stat['total']:<6} | {stat['acc1']:>6.1f}%  | {stat['acc2']:>6.1f}%  | {stat['acc3']:>6.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
