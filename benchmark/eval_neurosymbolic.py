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
        save_json,
    )
    from core.analyzer import FullGameStateAnalyzer


class MahjongNeuroSymbolicModel:
    def __init__(self,
                 base_model_path: str,
                 decision_lora_path: Optional[str] = None,
                 explanation_lora_path: Optional[str] = None):

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
                model.load_adapter(explanation_lora_path,
                                   adapter_name="explanation_adapter")
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

        print("--- Initializing mahjong calculator (Python) ---")
        self.analyzer = FullGameStateAnalyzer()
        print("--- Analyzer initialized ---")

    def _build_prompt(self, instruction: str, user_input: str) -> str:
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": user_input}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def predict(self, raw_input_text: str, run_explanation: bool = True) -> Tuple[List[str], str, str]:
        """
        Run inference (benchmark, robust extraction).
        Returns: (Top3 decisions, explanation, Stage2 prompt preview)
        """

        top3_discards = []

        DECISION_INSTRUCTION = "You are an expert Japanese Mahjong AI. Analyze the following game state and determine which tile to discard."
        decision_prompt = self._build_prompt(
            DECISION_INSTRUCTION, raw_input_text)

        if self.has_decision_adapter:
            self.model.set_adapter("decision_adapter")

        inputs = self.tokenizer(
            decision_prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=64, num_beams=5, num_return_sequences=3,
                    eos_token_id=self.stop_token_id, pad_token_id=self.tokenizer.pad_token_id, early_stopping=True
                )

            raw_candidates = []
            input_len = inputs.input_ids.shape[1]

            for output_ids in outputs:
                decoded_text = self.tokenizer.decode(
                    output_ids[input_len:], skip_special_tokens=True).strip()

                tile = extract_generated_discard(decoded_text)

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
        model_thoughts = "N/A"
        formatted_calc_data = ""
        explanation_input_final = ""

        if (
            run_explanation
            and self.has_explanation_adapter
            and best_discard not in ["INVALID", "Error"]
        ):
            try:
                symbolic_result = self.analyzer.analyze(raw_input_text)
                formatted_calc_data = format_comparative_data(
                    symbolic_result, best_discard)
            except Exception as e:
                formatted_calc_data = f"（Calculator error: {e}）"

            self.model.set_adapter("explanation_adapter")

            explanation_input_final = (
                f"【牌局详情】:\n{raw_input_text}\n\n"
                f"【客观数据与对比】\n{formatted_calc_data}\n\n"
                f"【已定决策】:\n切 {best_discard}"
            )

            EXPLANATION_INSTRUCTION = "你是一个严谨的麻将战术分析师。请基于牌局和客观计算数据（含切牌对比表），分析决策合理性。"

            explanation_prompt = self._build_prompt(
                EXPLANATION_INSTRUCTION, explanation_input_final)
            explanation_inputs = self.tokenizer(
                explanation_prompt, return_tensors="pt").to(self.model.device)

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

                model_thoughts = extract_thoughts(generated_text)

            except Exception as e:
                model_thoughts = f"Stage 2 Error: {e}"

            return top3_discards, model_thoughts, explanation_input_final

        return top3_discards, "Skipped Stage 2", ""


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the full NeSy-Mahjong pipeline")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--decision_lora_path", type=str, required=True)
    parser.add_argument("--explanation_lora_path", type=str, required=True)
    parser.add_argument("--split_dir", type=str, default="split_test")
    parser.add_argument("--output_dir", type=str,
                        default="results/neurosymbolic")
    parser.add_argument("--explanation_limit", type=int, default=40)
    args = parser.parse_args()

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
    print("Starting the full NeSy-Mahjong benchmark")
    print("Pipeline: Stage 1 -> calculator -> Stage 2")
    print("=" * 60 + "\n")

    for bucket_file in TEST_BUCKETS:
        file_path = os.path.join(args.split_dir, bucket_file)
        test_samples = load_jsonl(file_path)

        if not test_samples:
            print(f"Skipping missing or empty bucket: {bucket_file}")
            continue

        bucket_results = []
        correct_top1 = 0
        correct_top3 = 0
        total_count = 0

        print(f"Evaluating {bucket_file} ({len(test_samples)} samples)")

        for index, sample in enumerate(tqdm(test_samples, desc=f"Testing {bucket_file}")):
            raw_input = sample.get("input")
            ground_truth = get_ground_truth_discard(sample.get("output"))

            if not raw_input or ground_truth not in VALID_TILES:
                continue

            try:
                top3_discards, model_thoughts, prompt_used = model.predict(
                    raw_input, run_explanation=index < args.explanation_limit
                )

                best_decision = top3_discards[0]
                is_correct = (best_decision == ground_truth)
                is_correct_top3 = ground_truth in top3_discards
                if is_correct:
                    correct_top1 += 1
                if is_correct_top3:
                    correct_top3 += 1
                total_count += 1

                bucket_results.append({
                    "raw_input": raw_input,
                    "ground_truth": ground_truth,
                    "model_decision": best_decision,
                    "model_thoughts": model_thoughts,
                    "stage2_prompt": prompt_used,
                    "is_correct": is_correct,
                    "is_correct_top3": is_correct_top3,
                })

            except Exception as e:
                print(f"Sample Error: {e}")

        acc = (correct_top1 / total_count * 100) if total_count > 0 else 0
        acc_top3 = (correct_top3 / total_count * 100) if total_count > 0 else 0

        result_filename = f"result_{bucket_file.replace('.jsonl', '.json')}"
        save_path = os.path.join(args.output_dir, result_filename)
        save_json(bucket_results, save_path)

        summary_stats.append({
            "bucket": bucket_file, "total": total_count,
            "acc": acc, "acc_top3": acc_top3,
        })
        print(
            f"Finished {bucket_file}: Top-1 {acc:.1f}% | Top-3 {acc_top3:.1f}%")

    print("\n" + "=" * 40)
    for stat in summary_stats:
        print(
            f"{stat['bucket']:<25} | Top-1 {stat['acc']:>6.2f}%"
            f" | Top-3 {stat['acc_top3']:>6.2f}%"
        )
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
