import re
from collections import Counter
from .engine import RiichiCalculator


class MahjongAnalyzer:
    def __init__(self):
        self.calculator = RiichiCalculator()
        self.tile_pattern = re.compile(r"([0-9][mps]|Wh|[ESWNGR])\*?")
        self.meld_pattern = re.compile(r"\[([a-zA-Z]+) ([^\]]+)\]")

    def _normalize_tile(self, tile_str):
        """Normalize tile: remove riichi marker *, convert red 5 (0) to 5"""
        if not tile_str:
            return None
        clean_tile = tile_str.replace('*', '')
        if not clean_tile:
            return None
        if clean_tile[0] == '0' and clean_tile[-1] in 'mps':
            return '5' + clean_tile[-1]
        return clean_tile

    def _extract_tiles_from_text(self, text):
        if not text:
            return []
        raw_tiles = self.tile_pattern.findall(text)
        return [self._normalize_tile(t) for t in raw_tiles if t]

    def _get_visible_counts(self, input_json):
        """Count visible tiles (dora indicators, discards, melds, hand)"""
        counter = Counter()
        dora_match = re.search(r"Dora Indicators: (.*)", input_json)
        if dora_match:
            counter.update(self._extract_tiles_from_text(dora_match.group(1)))
        table_status_match = re.search(
            r"--- Table Status ---(.*)", input_json, re.DOTALL)
        if table_status_match:
            table_text = table_status_match.group(1)
            counter.update(self._extract_tiles_from_text(table_text))
            # A claimed discard appears once in a river and once in its meld.
            # Remove the duplicate occurrence while retaining all visible tiles.
            for meld_type, meld_body in self.meld_pattern.findall(table_text):
                if meld_type == "Ankan":
                    continue
                called_match = re.search(
                    r"<([0-9][mps]|Wh|[ESWNGR])|([0-9][mps]|Wh|[ESWNGR])<",
                    meld_body,
                )
                if called_match:
                    called_tile = called_match.group(
                        1) or called_match.group(2)
                    normalized = self._normalize_tile(called_tile)
                    if counter[normalized] > 0:
                        counter[normalized] -= 1
        hand_match = re.search(r"Hand: (.*?)(\n|$)", input_json)
        if hand_match:
            counter.update(self._extract_tiles_from_text(hand_match.group(1)))
        drawn_match = re.search(r"Drawn Tile: (.*?)(\n|$)", input_json)
        hand_tiles = self._extract_tiles_from_text(
            hand_match.group(1)) if hand_match else []
        if drawn_match and len(hand_tiles) % 3 == 1:
            counter.update(self._extract_tiles_from_text(drawn_match.group(1)))
        return counter

    def _get_open_meld_count(self, input_json, concealed_tile_count):
        pov_match = re.search(r"POV: Player (\d+)", input_json)
        if pov_match:
            player_pattern = rf"Player {pov_match.group(1)} .*?(?=\nPlayer \d+ |\Z)"
            player_match = re.search(player_pattern, input_json, re.DOTALL)
            if player_match:
                melds_match = re.search(
                    r"Melds: (.*?)(\n|$)", player_match.group(0))
                if melds_match:
                    return min(4, len(self.meld_pattern.findall(melds_match.group(1))))
        # A complete decision hand has 14, 11, 8, 5, or 2 concealed tiles.
        return max(0, min(4, (14 - concealed_tile_count) // 3))

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
        open_melds = self._get_open_meld_count(
            input_json, len(calc_hand_tiles))

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
                shanten = self.calculator.calc_shanten(temp_hand, open_melds)

                ukeire_data = self.calculator.calc_ukeire(
                    temp_hand, open_melds)

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
                results_list.append(
                    {"discard_tile": discard_candidate, "error": str(e)})

        results_list.sort(key=lambda x: (
            x.get('shanten', 99), -x.get('ukeire', 0)))

        return results_list
