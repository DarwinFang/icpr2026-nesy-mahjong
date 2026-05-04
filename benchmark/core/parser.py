import re
import copy
from collections import Counter
from .engine import RiichiCalculator


class MahjongAnalyzer:
    def __init__(self):
        self.calculator = RiichiCalculator()
        self.tile_pattern = re.compile(r"([0-9][mps]|Wh|[ESWNGR])\*?")
        self.meld_pattern = re.compile(r"\[([a-zA-Z]+) ([^\]]+)\]")

    def _normalize_tile(self, tile_str):
        """Normalize tile: remove riichi marker *, convert red 5 (0) to 5"""
        if not tile_str: return None
        clean_tile = tile_str.replace('*', '')
        if not clean_tile: return None
        if clean_tile[0] == '0' and clean_tile[-1] in 'mps':
            return '5' + clean_tile[-1]
        return clean_tile

    def _extract_tiles_from_text(self, text):
        if not text: return []
        raw_tiles = self.tile_pattern.findall(text)
        return [self._normalize_tile(t) for t in raw_tiles if t]

    def _get_visible_counts(self, input_json):
        """Count visible tiles (dora indicators, discards, melds, hand)"""
        counter = Counter()
        dora_match = re.search(r"Dora Indicators: (.*)", input_json)
        if dora_match:
            counter.update(self._extract_tiles_from_text(dora_match.group(1)))
        table_status_match = re.search(r"--- Table Status ---(.*)", input_json, re.DOTALL)
        if table_status_match:
            counter.update(self._extract_tiles_from_text(table_status_match.group(1)))
        hand_match = re.search(r"Hand: (.*?)(\n|$)", input_json)
        if hand_match:
            counter.update(self._extract_tiles_from_text(hand_match.group(1)))
        drawn_match = re.search(r"Drawn Tile: (.*?)(\n|$)", input_json)
        if drawn_match:
            counter.update(self._extract_tiles_from_text(drawn_match.group(1)))
        return counter

    def analyze_game_state(self, input_json):
        """
        Return sorted discard suggestions
        """
        hand_line_match = re.search(r"Hand: (.*?)(\n|$)", input_json)
        drawn_tile_match = re.search(r"Drawn Tile: (.*?)(\n|$)", input_json)

        if not hand_line_match:
            return [{"error": "Could not parse Hand"}]

        raw_hand_tiles = self.tile_pattern.findall(hand_line_match.group(1))

        if drawn_tile_match:
            drawn_tile = drawn_tile_match.group(1).replace('*', '')
            if len(raw_hand_tiles) % 3 == 1:
                raw_hand_tiles.append(drawn_tile)

        calc_hand_tiles = [t.replace('*', '') for t in raw_hand_tiles]

        visible_counter = self._get_visible_counts(input_json)

        results_list = []
        unique_discards = sorted(list(set(calc_hand_tiles)))

        for discard_candidate in unique_discards:
            temp_hand = list(calc_hand_tiles)
            if discard_candidate in temp_hand:
                temp_hand.remove(discard_candidate)
            else:
                continue

            try:
                shanten = self.calculator.calc_shanten(temp_hand)

                ukeire_data = self.calculator.calc_ukeire(temp_hand)

                if 'ukeire' not in ukeire_data:
                    continue

                naive_ukeire_dict = ukeire_data['ukeire']
                actual_ukeire_details = {}
                total_actual_ukeire = 0

                for tile, _ in naive_ukeire_dict.items():
                    norm_tile = self._normalize_tile(tile)
                    seen_count = visible_counter[norm_tile]
                    left_count = max(0, 4 - seen_count)

                    if left_count > 0:
                        actual_ukeire_details[tile] = left_count
                        total_actual_ukeire += left_count

                results_list.append({
                    "discard_tile": discard_candidate,
                    "shanten": shanten,
                    "ukeire": total_actual_ukeire,
                    "details": actual_ukeire_details
                })

            except Exception as e:
                results_list.append({"discard_tile": discard_candidate, "error": str(e)})

        results_list.sort(key=lambda x: (x.get('shanten', 99), -x.get('ukeire', 0)))

        return results_list


if __name__ == "__main__":
    analyzer = MahjongAnalyzer()

    data_sample = """Game: 南4局, 0 Honba, 0 Riichi Sticks, 35 Tiles Left
Oya (Dealer): Player 3
Dora Indicators: 6s
--- Your Status ---
POV: Player 0 (Score: 38200) (South)
Hand: 5m 6m 6p 7p 5s 5s 6s 7s 7s 8s 8s 9s Wh E
Drawn Tile: E
--- Table Status ---
Player 0 (Score: 38200) (South) (You):
  Discards: 9p 9m 9p E Wh 2s 4s 2m W
Player 1 (Score: 31200) (West) (Melds: [Pon 9p<9p 9p from P2]):
  Discards: 1p R 8s"""

    print("--- Analyzing Case 1 (Sorted) ---")
    results = analyzer.analyze_game_state(data_sample)

    for res in results[:3]:
        print(res)

    data_sample_3 = """Game: 南1局...
--- Your Status ---
POV: Player 0 (Score: 21100) (East)
Hand: 6m 7m 3p 5p 7p 8p 8p 3s 4s 6s 1p
Drawn Tile: 1p
--- Table Status ---
Player 0 (Score: 21100) (East) (Oya) (You) (Melds: [Ankan N N N N from P0]):
  Discards: 9m 1m G E E
Player 3 (Score: 17200) (North) (Riichi):
  Discards: G 9p R 7p 5m*"""

    print("\n--- Analyzing Case 3 (Sorted) ---")
    results_3 = analyzer.analyze_game_state(data_sample_3)
    for res in results_3:
        print(f"Discard {res['discard_tile']}: Shanten={res['shanten']}, Ukeire={res['ukeire']}")
