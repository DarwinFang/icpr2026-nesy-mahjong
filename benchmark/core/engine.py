import copy


class RiichiCalculator:
    def __init__(self):
        self.suit_map = {}

        self.TILE_NAMES = [
            [f'{i}m' for i in range(1, 10)],
            [f'{i}p' for i in range(1, 10)],
            [f'{i}s' for i in range(1, 10)],
            ['E', 'S', 'W', 'N', 'Wh', 'G', 'R'],
        ]

        self.HONOR_MAP = {
            'E': 0, 'S': 1, 'W': 2, 'N': 3,
            'Wh': 4, 'G': 5, 'R': 6
        }

        self.kokushi_honour = [1] * 7
        self.kokushi_plain = [1, 0, 0, 0, 0, 0, 0, 0, 1]

    def _tiles_to_hand(self, tiles_arr):
        """
        Convert string list to 4x9 (or 4x7) 2D array
        Supports: '1m', '0m'(red 5), 'E', 'Wh' etc.
        """
        hand = [
            [0] * 9,
            [0] * 9,
            [0] * 9,
            [0] * 7
        ]

        for tile in tiles_arr:
            if not tile: continue

            if tile[-1] in 'mps':
                try:
                    suit_char = tile[-1]
                    rank_char = tile[:-1]
                    rank = int(rank_char)

                    if rank == 0:
                        idx = 4
                    else:
                        idx = rank - 1

                    suit_map = {'m': 0, 'p': 1, 's': 2}
                    if 0 <= idx <= 8:
                        hand[suit_map[suit_char]][idx] += 1
                except (ValueError, KeyError):
                    continue

            elif tile in self.HONOR_MAP:
                idx = self.HONOR_MAP[tile]
                hand[3][idx] += 1

        return hand

    def _hand_length(self, hand):
        return sum(sum(suit) for suit in hand)

    def _suit_to_str(self, suit, is_honor):
        if is_honor:
            counts = [0, 0, 0, 0]
            for c in suit:
                if c > 0: counts[c - 1] += 1
            return "z" + "".join(map(str, counts))
        else:
            res = []
            for i, c in enumerate(suit):
                res.extend([str(i + 1)] * c)
            return "".join(res)

    def _cal_optimal_suit_combination(self, suit_in, is_honor=False):
        suit_str = self._suit_to_str(suit_in, is_honor)
        if suit_str in self.suit_map:
            return self.suit_map[suit_str]

        suit = list(suit_in)
        current_state = [0, 0, 0]
        max_res = [0, 0, 0]
        residuals = []

        def remove_taatsus(idx=0):
            if current_state[1] > max_res[1]:
                max_res[1] = current_state[1]
            if current_state[2] > 0 and current_state[1] > max_res[2]:
                max_res[2] = current_state[1]

            if idx >= len(suit): return
            while idx < len(suit) and suit[idx] == 0:
                idx += 1
                if idx >= len(suit): return

            if suit[idx] >= 2:
                current_state[1] += 1;
                current_state[2] += 1
                suit[idx] -= 2
                remove_taatsus(idx)
                suit[idx] += 2
                current_state[1] -= 1;
                current_state[2] -= 1

            if not is_honor:
                if idx + 2 < 9 and suit[idx + 2] > 0:
                    current_state[1] += 1
                    suit[idx] -= 1;
                    suit[idx + 2] -= 1
                    remove_taatsus(idx)
                    suit[idx] += 1;
                    suit[idx + 2] += 1
                    current_state[1] -= 1

                if idx + 1 < 9 and suit[idx + 1] > 0:
                    current_state[1] += 1
                    suit[idx] -= 1;
                    suit[idx + 1] -= 1
                    remove_taatsus(idx)
                    suit[idx] += 1;
                    suit[idx + 1] += 1
                    current_state[1] -= 1

            remove_taatsus(idx + 1)

        def remove_groups(idx=0):
            if current_state[0] > max_res[0]:
                max_res[0] = current_state[0]
                residuals.clear()
            if current_state[0] == max_res[0]:
                residuals.append(list(suit))

            if idx >= len(suit): return
            while idx < len(suit) and suit[idx] == 0:
                idx += 1
                if idx >= len(suit): return

            if suit[idx] >= 3:
                current_state[0] += 1
                suit[idx] -= 3
                remove_groups(idx)
                suit[idx] += 3
                current_state[0] -= 1

            if not is_honor and idx + 2 < 9 and suit[idx + 1] > 0 and suit[idx + 2] > 0:
                current_state[0] += 1
                suit[idx] -= 1;
                suit[idx + 1] -= 1;
                suit[idx + 2] -= 1
                remove_groups(idx)
                suit[idx] += 1;
                suit[idx + 1] += 1;
                suit[idx + 2] += 1
                current_state[0] -= 1

            remove_groups(idx + 1)

        remove_groups(0)
        temp_groups_count = max_res[0]
        for res_suit in residuals:
            suit = res_suit
            current_state = [temp_groups_count, 0, 0]
            remove_taatsus(0)
            if sum(suit) <= 1 and current_state[2] > 0:
                break

        self.suit_map[suit_str] = tuple(max_res)
        return tuple(max_res)


    def _cal_shanten_menzu(self, hand):
        stats = [
            self._cal_optimal_suit_combination(hand[0], False),
            self._cal_optimal_suit_combination(hand[1], False),
            self._cal_optimal_suit_combination(hand[2], False),
            self._cal_optimal_suit_combination(hand[3], True)
        ]
        target_sets = 4
        total_groups = sum(s[0] for s in stats)
        total_taatsus = sum(s[1] for s in stats)
        deficit = target_sets - total_groups

        def calculate_shanten(def_, taatsus, pair_exists):
            if taatsus < def_ + 1:
                return 2 * def_ - taatsus
            else:
                return def_ - (1 if pair_exists else 0)

        shanten = calculate_shanten(deficit, total_taatsus, False)
        for i in range(4):
            if stats[i][2] > 0:
                current_taatsus = total_taatsus - stats[i][1] + stats[i][2]
                s = calculate_shanten(deficit, current_taatsus, True)
                shanten = min(shanten, s)
        return shanten

    def _cal_shanten_chiitoi(self, hand):
        pairs = 0
        unique_tiles = 0
        for suit in hand:
            for count in suit:
                if count > 0: unique_tiles += 1
                if count >= 2: pairs += 1
        shanten = 6 - pairs
        if unique_tiles < 7: shanten += (7 - unique_tiles)
        return shanten

    def _cal_shanten_kokushi(self, hand):
        has_pair = False
        unique_match = 0
        for i in range(7):
            if hand[3][i] > 0:
                unique_match += 1
                if hand[3][i] >= 2: has_pair = True
        for s in range(3):
            if hand[s][0] > 0:
                unique_match += 1
                if hand[s][0] >= 2: has_pair = True
            if hand[s][8] > 0:
                unique_match += 1
                if hand[s][8] >= 2: has_pair = True
        return 13 - unique_match - (1 if has_pair else 0)


    def calc_shanten(self, hand_or_tiles):
        if isinstance(hand_or_tiles, list) and isinstance(hand_or_tiles[0], str):
            hand = self._tiles_to_hand(hand_or_tiles)
        else:
            hand = hand_or_tiles
        return min(self._cal_shanten_menzu(hand),
                   self._cal_shanten_chiitoi(hand),
                   self._cal_shanten_kokushi(hand))

    def calc_ukeire(self, hand_or_tiles):
        if isinstance(hand_or_tiles, list) and isinstance(hand_or_tiles[0], str):
            hand = self._tiles_to_hand(hand_or_tiles)
        else:
            hand = hand_or_tiles

        tile_count = self._hand_length(hand)
        if tile_count % 3 == 1:
            return self._ukeire_draw(hand)
        else:
            return self._ukeire_discard(hand)

    def _ukeire_draw(self, hand):
        current_shanten = self.calc_shanten(hand)
        valid_tiles = {}
        total_count = 0

        for i in range(4):
            for j in range(len(hand[i])):
                if hand[i][j] >= 4: continue

                hand[i][j] += 1
                new_shanten = self.calc_shanten(hand)
                hand[i][j] -= 1

                if new_shanten < current_shanten:
                    tile_name = self.TILE_NAMES[i][j]
                    remaining = 4 - hand[i][j]
                    valid_tiles[tile_name] = remaining
                    total_count += remaining

        return {
            'shanten': current_shanten,
            'ukeire': valid_tiles,
            'total_ukeire': total_count
        }

    def _ukeire_discard(self, hand):
        current_shanten = self.calc_shanten(hand)
        normal_discard = {}
        receding_discard = {}

        for i in range(4):
            for j in range(len(hand[i])):
                if hand[i][j] > 0:
                    tile_name = self.TILE_NAMES[i][j]
                    hand[i][j] -= 1
                    result = self._ukeire_draw(hand)
                    hand[i][j] += 1

                    if result['shanten'] > current_shanten:
                        if tile_name not in receding_discard:
                            receding_discard[tile_name] = result['ukeire']
                    else:
                        if tile_name not in normal_discard:
                            normal_discard[tile_name] = result['ukeire']

        return {
            'shanten': current_shanten,
            'normal_discard': normal_discard,
            'receding_discard': receding_discard
        }


if __name__ == "__main__":
    calculator = RiichiCalculator()

    tiles = [
        '1m', '2m', '3m',
        '6p', '7p', '7p', '7p', '8p', '8p',
        '0s', '5s',
        'S', 'Wh'
    ]

    print(f"Hand: {tiles}")

    shanten = calculator.calc_shanten(tiles)
    print(f"Current Shanten: {shanten}")

    ukeire_res = calculator.calc_ukeire(tiles)
    print("Ukeire Result:")
    print(ukeire_res['ukeire'])
    print(f"Total tiles: {ukeire_res['total_ukeire']}")

    tiles_14 = tiles + ['Wh']
    print(f"\n--- Simulate 14 tiles (Discard mode, Draw: Wh) ---")
    discard_res = calculator.calc_ukeire(tiles_14)
    print(f"Normal Discards (Maintain Shanten {discard_res['shanten']}):")
    for tile, waiting in discard_res['normal_discard'].items():
        print(f"Discard {tile} -> Wait for: {list(waiting.keys())}")
