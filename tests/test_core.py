import json
import unittest

from benchmark.common import (
    extract_generated_discard,
    extract_thoughts,
    format_comparative_data,
    get_ground_truth_discard,
)
from benchmark.core.analyzer import FullGameStateAnalyzer
from benchmark.core.engine import RiichiCalculator
from benchmark.core.parser import MahjongAnalyzer


class GroundTruthTests(unittest.TestCase):
    def test_reads_all_supported_output_formats(self):
        self.assertEqual(get_ground_truth_discard(
            {"discard_tile": "5s*"}), "5s")
        self.assertEqual(get_ground_truth_discard(
            '{"discard_tile": "Wh"}'), "Wh")
        self.assertEqual(get_ground_truth_discard("9m"), "9m")
        self.assertEqual(get_ground_truth_discard(None), "")

    def test_white_dragon_is_not_truncated_to_west(self):
        self.assertEqual(extract_generated_discard(
            "$ Wh", prefer_dollar=True), "Wh")
        self.assertEqual(
            extract_generated_discard('{"discard_tile": "Wh"}'), "Wh"
        )

    def test_extracts_json_explanation(self):
        self.assertEqual(
            extract_thoughts('{"thoughts": "牌效和防守兼顾。"}'),
            "牌效和防守兼顾。",
        )

    def test_stage2_format_matches_training_schema(self):
        formatted = format_comparative_data(
            {
                "process_estimates": "中巡",
                "tenpai_estimates": {0: "High", 1: "Low"},
                "tile_analysis": {
                    "9m": {
                        "shanten": 2,
                        "ukeire": 16,
                        "safety_analysis": "P0现物",
                    }
                },
            },
            "9m",
        )
        self.assertIn("全局形势：【中巡】听牌概率：P0: High, P1: Low", formatted)
        self.assertIn("- [切9m]: 2向听, 进16张 | P0现物 👈(实战决策)", formatted)


class CalculatorTests(unittest.TestCase):
    def test_open_meld_reduces_required_concealed_sets(self):
        calculator = RiichiCalculator()
        concealed = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E"]
        self.assertEqual(calculator.calc_shanten(concealed, open_melds=1), 0)
        self.assertGreater(calculator.calc_shanten(concealed), 0)

    def test_empty_input_is_handled(self):
        calculator = RiichiCalculator()
        self.assertIsInstance(calculator.calc_shanten([]), int)
        self.assertIn("shanten", calculator.calc_ukeire([]))


class StateParsingTests(unittest.TestCase):
    STATE = """Game: East 2, 0 Honba, 0 Riichi Sticks, 40 Tiles Left
Oya (Dealer): Player 0
Dora Indicators: 6s
--- Your Status ---
POV: Player 0 (Score: 25000) (East)
Hand: 1m 2m 3m 4p 5p 6p 7s 8s 9s E E 2p 2p 2p
Drawn Tile: 2p
--- Table Status ---
Player 0 (Score: 25000) (East) (You):
  Discards: 9p
Player 1 (Score: 25000) (South) (Melds: [Pon <9p 9p 9p from P0]):
  Discards: E
Player 2 (Score: 25000) (West):
  Discards:
Player 3 (Score: 25000) (North):
  Discards:
"""

    def test_drawn_tile_is_not_counted_twice(self):
        counts = MahjongAnalyzer()._get_visible_counts(self.STATE)
        self.assertEqual(counts["2p"], 3)

    def test_called_discard_is_not_counted_twice(self):
        counts = MahjongAnalyzer()._get_visible_counts(self.STATE)
        self.assertEqual(counts["9p"], 3)

    def test_full_analyzer_accepts_complete_hand(self):
        result = FullGameStateAnalyzer().analyze(self.STATE)
        self.assertIn("tile_analysis", result)
        self.assertTrue(result["tile_analysis"])


if __name__ == "__main__":
    unittest.main()
