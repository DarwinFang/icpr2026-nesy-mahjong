import json
import sys
import torch
import argparse
import os
import re
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Set

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from outlines.models.transformers import Transformers
from outlines.generator import Generator

VALID_TILES_list = [
    "1m", "2m", "3m", "4m", "5m", "0m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "0p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "0s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "Wh", "G", "R"
]
VALID_TILES: Set[str] = set(VALID_TILES_list)


class MahjongV5Explanation(BaseModel):
    """V5 explanation model output format (LoRA B)"""
    thoughts: str = Field(
        description="My step-by-step reasoning for this decision."
    )


class MahjongTop3Model:
    """
    Model supporting Top-3 beam search decisions and V9 explanation generation
    """

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
                model,
                decision_lora_path,
                adapter_name="decision_adapter"
            )
            self.has_decision_adapter = True
        else:
            print("--- ⚠️ No decision LoRA provided; base model will decide ---")

        if explanation_lora_path:
            print(f"--- Loading explanation LoRA: {explanation_lora_path} ---")
            if self.has_decision_adapter:
                model.load_adapter(explanation_lora_path, adapter_name="explanation_adapter")
            else:
                model = PeftModel.from_pretrained(
                    model,
                    explanation_lora_path,
                    adapter_name="explanation_adapter"
                )
            self.has_explanation_adapter = True
        else:
            print("--- ⚠️ No explanation LoRA provided; skipping explanation ---")

        model.eval()
        self.model = model
        if isinstance(self.tokenizer.eos_token_id, list):
            self.stop_token_id = self.tokenizer.eos_token_id[0]
        else:
            self.stop_token_id = self.tokenizer.eos_token_id

        self.json_pattern = re.compile(r':\s*"([^"]+)"')
        self.start_pattern = re.compile(r'^([1-9][mps]|0[mps]|[ESWN]|Wh|G|R)')

    def _build_prompt(self, instruction: str, user_input: str) -> str:
        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": user_input}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def predict(self,
                decision_instruction: str,
                game_input: str,
                explanation_instruction: str
                ) -> Tuple[List[str], str]:
        """
        Run inference.
        Returns: ([Top1, Top2, Top3 discards], Top1 explanation)
        """

        top3_discards = []

        if self.has_decision_adapter:
            self.model.set_adapter("decision_adapter")

        decision_prompt = self._build_prompt(decision_instruction, game_input)
        inputs = self.tokenizer(decision_prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=5,
                    num_return_sequences=3,
                    eos_token_id=self.stop_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    early_stopping=True
                )

            raw_candidates = []
            input_len = inputs.input_ids.shape[1]

            for output_ids in outputs:
                generated_ids = output_ids[input_len:]
                decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                tile_candidate = None

                json_match = self.json_pattern.search(decoded_text)
                if json_match:
                    tile_candidate = json_match.group(1)
                else:
                    start_match = self.start_pattern.match(decoded_text)
                    if start_match:
                        tile_candidate = start_match.group(1)

                if tile_candidate:
                    clean_tile = tile_candidate.replace('*', '').strip()
                    if clean_tile in VALID_TILES:
                        raw_candidates.append(clean_tile)

            seen = set()
            for x in raw_candidates:
                if x not in seen:
                    top3_discards.append(x)
                    seen.add(x)

            while len(top3_discards) < 3:
                top3_discards.append("INVALID")

        except Exception as e:
            print(f"Beam search failed: {e}")
            top3_discards = ["Error", "Error", "Error"]

        model_thoughts = "N/A"
        best_discard = top3_discards[0]

        if self.has_explanation_adapter and best_discard not in ["INVALID", "Error", "N/A"]:
            self.model.set_adapter("explanation_adapter")

            outlines_model_B = Transformers(self.model, self.tokenizer)
            explanation_generator = Generator(outlines_model_B, MahjongV5Explanation)

            explanation_input = (
                f"【Game state】:\n{game_input}\n\n"
                f"【Decision】:\ndiscard  {best_discard}"
            )

            explanation_prompt = self._build_prompt(explanation_instruction, explanation_input)

            try:
                explanation_json_str = explanation_generator(
                    explanation_prompt,
                    max_new_tokens=1024,
                    eos_token_id=self.stop_token_id
                )

                last_brace_index = explanation_json_str.rfind('}')
                if last_brace_index != -1:
                    clean_json_str = explanation_json_str[:last_brace_index + 1]
                else:
                    clean_json_str = explanation_json_str.strip()

                explanation_obj = MahjongV5Explanation.model_validate_json(clean_json_str)
                model_thoughts = explanation_obj.thoughts
            except Exception:
                model_thoughts = "Generation Error"

        return top3_discards, model_thoughts


def load_test_samples(filename: str) -> List[Dict[str, Any]]:
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except:
                pass
    return samples


def save_results(results_data: List[Dict[str, Any]], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)


def get_ground_truth_discard(output_field: Any) -> str:
    if isinstance(output_field, str):
        try:
            return json.loads(output_field).get("discard_tile", output_field.strip())
        except:
            return output_field.strip()
    if isinstance(output_field, dict): return output_field.get("discard_tile", "N/A")
    return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Run Mahjong Top-3 Benchmark (V9-Pro Aligned)")
    parser.add_argument("--base_model_path", type=str, default="../llama3-8b")
    parser.add_argument("--decision_lora_path", type=str, default=None, help="Optional: Path to decision LoRA")
    parser.add_argument("--explanation_lora_path", type=str, default=None, help="Optional: Path to explanation LoRA")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--output_file", type=str, default="benchmark_top3_results.json")
    args = parser.parse_args()

    DECISION_INSTRUCTION = "You are a mahjong AI. Analyze the game state and output only your JSON decision."

    EXPLANATION_INSTRUCTION = "You are a rigorous mahjong analyst. Analyze this discard from efficiency, defense, and value perspectives."

    try:
        model = MahjongTop3Model(
            args.base_model_path,
            args.decision_lora_path,
            args.explanation_lora_path
        )
        test_samples = load_test_samples(args.test_file)
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    results_data = []
    correct_count_top1 = 0
    correct_count_top3 = 0
    total_count = 0

    desc = "Benchmarking"
    if args.decision_lora_path:
        desc += " [Dec: LoRA]"
    else:
        desc += " [Dec: Base]"
    if args.explanation_lora_path:
        desc += " [Exp: LoRA]"
    else:
        desc += " [Exp: None]"

    for sample in tqdm(test_samples, desc=desc):
        game_input = sample.get("input")
        ground_truth_output = sample.get("output")
        if not game_input or not ground_truth_output: continue

        ground_truth_discard = get_ground_truth_discard(ground_truth_output)
        if ground_truth_discard == "N/A": continue

        try:
            top3_discards, model_thoughts = model.predict(
                DECISION_INSTRUCTION, game_input, EXPLANATION_INSTRUCTION
            )

            is_correct_top1 = (top3_discards[0] == ground_truth_discard)
            if is_correct_top1: correct_count_top1 += 1

            is_correct_top3 = (ground_truth_discard in top3_discards)
            if is_correct_top3: correct_count_top3 += 1

            total_count += 1

            results_data.append({
                "input": game_input,
                "ground_truth_discard": ground_truth_discard,
                "model_top3": top3_discards,
                "model_thoughts": model_thoughts,
                "is_correct_top1": is_correct_top1,
                "is_correct_top3": is_correct_top3
            })
        except Exception as e:
            print(f"Error processing sample: {e}")

    if total_count > 0:
        save_results(results_data, args.output_file)
        acc_top1 = (correct_count_top1 / total_count) * 100
        acc_top3 = (correct_count_top3 / total_count) * 100

        print(f"\n✅ Benchmark Complete.")
        print(f"Total Samples: {total_count}")
        print(f"🏆 Top-1 Accuracy: {acc_top1:.2f}%")
        print(f"🥉 Top-3 Accuracy: {acc_top3:.2f}%")
        print(f"Results saved to {args.output_file}")
    else:
        print("\nNo valid samples processed.")


if __name__ == "__main__":
    main()
