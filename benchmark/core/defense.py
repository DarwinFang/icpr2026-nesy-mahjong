import re
from collections import Counter

HONOR_TILES = {'E', 'S', 'W', 'N', 'Wh', 'G', 'R'}


def normalize_tile(tile_str):
    """Normalize: 0m->5m, 5m*->5m, Wh->Wh"""
    if not tile_str:
        return None
    t = tile_str.replace('*', '')
    if t.startswith('0') and t[-1] in 'mps':
        return '5' + t[-1]
    return t


def get_tile_rank_suit(tile_str):
    """
    Parse tile number and suit.
    Honor tiles E,S,W,N,Wh,G,R as rank=0 suit=z
    """
    norm = normalize_tile(tile_str)
    if not norm:
        return 0, ''

    if norm in HONOR_TILES:
        return 0, 'z'

    suit = norm[-1]
    if suit in 'mps':
        try:
            return int(norm[:-1]), suit
        except ValueError:
            return 0, ''

    return 0, ''


def get_real_dora(dora_indicator):
    """Compute real dora"""
    if not dora_indicator:
        return None
    tile = normalize_tile(dora_indicator)

    if tile in HONOR_TILES:
        seq_wind = ['E', 'S', 'W', 'N']
        seq_drag = ['Wh', 'G', 'R']
        if tile in seq_wind:
            return seq_wind[(seq_wind.index(tile) + 1) % 4]
        if tile in seq_drag:
            return seq_drag[(seq_drag.index(tile) + 1) % 3]

    rank, suit = get_tile_rank_suit(tile)
    if suit in 'mps':
        new_val = 1 if rank == 9 else rank + 1
        return f"{new_val}{suit}"

    return None


class SafetyAnalyzer:
    def __init__(self, players_data, pov_hand, dealer_seat):
        """
        players_data: Dict[seat_id, {discards:[], melds:[], riichi:bool, riichi_tile_idx:int}]
        pov_hand: List[str]
        dealer_seat: int
        """
        self.players = players_data
        self.pov_hand = pov_hand
        self.dealer_seat = dealer_seat

        self.table_counts = Counter()
        self.total_counts = Counter()
        self._init_counts()

    def _init_counts(self):
        """Count visible tiles"""
        for t in self.pov_hand:
            norm = normalize_tile(t)
            self.total_counts[norm] += 1

        for p_data in self.players.values():
            for d in p_data['discards']:
                norm = normalize_tile(d)
                self.table_counts[norm] += 1
                self.total_counts[norm] += 1
            for m_str in p_data['melds']:
                tiles = re.findall(r"([0-9][mps]|Wh|[ESWNGR])", m_str)
                if not m_str.startswith("[Ankan"):
                    called_match = re.search(
                        r"<([0-9][mps]|Wh|[ESWNGR])|([0-9][mps]|Wh|[ESWNGR])<",
                        m_str,
                    )
                    if called_match:
                        called_tile = called_match.group(
                            1) or called_match.group(2)
                        for index, tile in enumerate(tiles):
                            if normalize_tile(tile) == normalize_tile(called_tile):
                                tiles.pop(index)
                                break
                for t in tiles:
                    norm = normalize_tile(t)
                    self.total_counts[norm] += 1

    def check_reach_genbutsu_strict(self, tile, riichi_seat, target_seat):
        norm_tile = normalize_tile(tile)
        riichi_data = self.players[riichi_seat]
        target_data = self.players[target_seat]
        target_discards = target_data['discards']

        if riichi_seat == target_seat:
            for d in target_discards:
                if normalize_tile(d) == norm_tile:
                    return True
            return False

        x = riichi_data['riichi_tile_idx']
        if x == -1:
            return False
        x += 1

        y = 0
        for seat, p_data in self.players.items():
            if seat == riichi_seat:
                continue
            for m_str in p_data['melds']:
                if f"from P{riichi_seat}" in m_str:
                    y += 1

        z = x + y

        t_B = len(target_data['melds'])

        dist_riichi = (riichi_seat - self.dealer_seat) % 4
        dist_target = (target_seat - self.dealer_seat) % 4

        threshold = 0
        if dist_target > dist_riichi:
            threshold = z + t_B
        else:
            threshold = z + t_B + 1

        for i in range(len(target_discards) - 1, -1, -1):
            d_obj = target_discards[i]
            if normalize_tile(d_obj) == norm_tile:
                idx_1based = i + 1
                if idx_1based >= threshold:
                    return True

        return False

    def get_suji_safety(self, tile, target_seat):
        rank, suit = get_tile_rank_suit(tile)
        if suit == 'z' or rank == 0 or not suit:
            return None

        target_discards = [normalize_tile(
            d) for d in self.players[target_seat]['discards']]

        def has_cut(r): return f"{r}{suit}" in target_discards

        reasons = []

        if rank == 1 and has_cut(4):
            reasons.append("筋(现4)")
        if rank == 2 and has_cut(5):
            reasons.append("筋(现5)")
        if rank == 3 and has_cut(6):
            reasons.append("筋(现6)")
        if rank == 7 and has_cut(4):
            reasons.append("筋(现4)")
        if rank == 8 and has_cut(5):
            reasons.append("筋(现5)")
        if rank == 9 and has_cut(6):
            reasons.append("筋(现6)")

        if rank == 4:
            c1, c7 = has_cut(1), has_cut(7)
            if c1 and c7:
                reasons.append("双筋(现1/7)")
            elif c1 or c7:
                reasons.append("半筋(现1或7)")
        if rank == 5:
            c2, c8 = has_cut(2), has_cut(8)
            if c2 and c8:
                reasons.append("双筋(现2/8)")
            elif c2 or c8:
                reasons.append("半筋(现2或8)")
        if rank == 6:
            c3, c9 = has_cut(3), has_cut(9)
            if c3 and c9:
                reasons.append("双筋(现3/9)")
            elif c3 or c9:
                reasons.append("半筋(现3或9)")

        return reasons if reasons else None

    def get_kabe_safety_info(self, tile):
        rank, suit = get_tile_rank_suit(tile)
        if suit == 'z' or rank == 0 or not suit:
            return []

        reasons = []
        def get_count(r): return self.total_counts[f"{r}{suit}"]

        if rank == 1 and get_count(2) >= 4:
            reasons.append("No-Chance(2断)")
        if rank in [1, 2] and get_count(3) >= 4:
            reasons.append("No-Chance(3断)")
        if rank in [2, 3] and get_count(4) >= 4:
            reasons.append("No-Chance(4断)")
        if rank in [7, 8] and get_count(6) >= 4:
            reasons.append("No-Chance(6断)")
        if rank in [8, 9] and get_count(7) >= 4:
            reasons.append("No-Chance(7断)")
        if rank == 9 and get_count(8) >= 4:
            reasons.append("No-Chance(8断)")

        if rank == 1 and get_count(2) == 3:
            reasons.append("One-Chance(2见3)")
        if rank in [1, 2] and get_count(3) == 3:
            reasons.append("One-Chance(3见3)")
        if rank in [2, 3] and get_count(4) == 3:
            reasons.append("One-Chance(4见3)")
        if rank in [7, 8] and get_count(6) == 3:
            reasons.append("One-Chance(6见3)")
        if rank in [8, 9] and get_count(7) == 3:
            reasons.append("One-Chance(7见3)")
        if rank == 9 and get_count(8) == 3:
            reasons.append("One-Chance(8见3)")

        return reasons

    def get_live_info(self, tile):
        norm = normalize_tile(tile)
        if self.total_counts[norm] >= 4:
            return "绝张"

        if self.table_counts[norm] == 0:
            return "生张"

        return ""

    def estimate_tenpai_prob(self, real_doras):
        """
        Compute tenpai probability via heuristic scoring.
        real_doras: list of strings (e.g., ['1m', '9p'])
        """
        results = {}

        for pid, p_data in self.players.items():

            if p_data['riichi']:
                results[pid] = "High (Riichi)"
                continue

            melds_count = len(p_data['melds'])
            prob = 0.10

            if melds_count == 4:
                prob = 1.0
            elif melds_count == 3:
                prob = 0.70
            elif melds_count == 2:
                prob = 0.40
            elif melds_count == 1:
                prob = 0.20

            current_turn = len(p_data['discards'])

            if current_turn > 6:
                prob += 0.15
            if current_turn > 12:
                prob += 0.25
                if melds_count == 0:
                    prob = max(prob, 0.45)

            late_discards = p_data['discards'][6:] if len(
                p_data['discards']) > 6 else []

            has_cut_aka = any(d.startswith('0') for d in late_discards)
            if has_cut_aka:
                prob += 0.20

            has_cut_dora = any(normalize_tile(
                d) in real_doras for d in late_discards)
            if has_cut_dora:
                prob += 0.15

            prob = min(prob, 0.95)

            level = "Low"
            if prob >= 0.8:
                level = "Very High"
            elif prob >= 0.5:
                level = "High"
            elif prob >= 0.25:
                level = "Medium"

            results[pid] = level

        return results
