import re
from .parser import MahjongAnalyzer
from .defense import SafetyAnalyzer, normalize_tile, get_real_dora


class FullGameStateAnalyzer:
    def __init__(self):
        self.efficiency_analyzer = MahjongAnalyzer()

    def _parse_full_state(self, input_text):
        """Parse table status in detail"""
        oya_match = re.search(r"Oya \(Dealer\): Player (\d+)", input_text)
        dealer_seat = int(oya_match.group(1)) if oya_match else 0

        tiles_match = re.search(r"(\d+) Tiles Left", input_text)
        tiles_left = int(tiles_match.group(1)) if tiles_match else 64

        dora_match = re.search(r"Dora Indicators: (.*)", input_text)
        dora_inds = dora_match.group(1).split() if dora_match else []

        hand_match = re.search(r"Hand: (.*?)(\n|$)", input_text)
        pov_hand_raw = hand_match.group(1).split() if hand_match else []
        drawn_match = re.search(r"Drawn Tile: (.*?)(\n|$)", input_text)
        if drawn_match and len(pov_hand_raw) % 3 == 1:
            pov_hand_raw.append(drawn_match.group(1).replace('*', ''))

        players_data = {}
        segments = re.split(r"(Player \d+ \(Score: -?\d+\).*?:)", input_text)

        for i in range(1, len(segments), 2):
            header = segments[i]
            content = segments[i + 1] if i + 1 < len(segments) else ""

            pid_match = re.search(r"Player (\d+)", header)
            if not pid_match:
                continue
            pid = int(pid_match.group(1))

            is_riichi = "(Riichi)" in header

            melds = []
            m_match = re.search(r"Melds: (.*?)(\n|$)", header + content)
            if m_match:
                melds = re.findall(r"\[.*?\]", m_match.group(1))

            discards = []
            riichi_idx = -1
            d_match = re.search(r"Discards: (.*?)(\n|$)", content)
            if d_match:
                raw_d = d_match.group(1).split()
                for idx, d in enumerate(raw_d):
                    discards.append(d)
                    if '*' in d:
                        riichi_idx = idx
                        is_riichi = True

            players_data[pid] = {
                'discards': discards,
                'melds': melds,
                'riichi': is_riichi,
                'riichi_tile_idx': riichi_idx
            }

        return {
            'dealer_seat': dealer_seat,
            'tiles_left': tiles_left,
            'dora_inds': dora_inds,
            'pov_hand': pov_hand_raw,
            'players': players_data
        }

    def analyze(self, input_json):
        eff_results_list = self.efficiency_analyzer.analyze_game_state(
            input_json)

        ctx = self._parse_full_state(input_json)

        safety = SafetyAnalyzer(
            ctx['players'], ctx['pov_hand'], ctx['dealer_seat'])

        pov_match = re.search(r"POV: Player (\d+)", input_json)
        pov_id = int(pov_match.group(1)) if pov_match else 0

        pov_discard_count = len(ctx['players'].get(
            pov_id, {}).get('discards', []))

        process_stage = "早巡"
        if pov_discard_count >= 12:
            process_stage = "晚巡"
        elif pov_discard_count >= 6:
            process_stage = "中巡"

        process_info = process_stage

        real_doras = [get_real_dora(d) for d in ctx['dora_inds']]
        tenpai_probs = safety.estimate_tenpai_prob(real_doras)

        risky_pids = []
        for pid, status in tenpai_probs.items():
            if pid == pov_id:
                continue
            if status in {"High (Riichi)", "Very High", "High"}:
                risky_pids.append(pid)

        analysis_dict = {}

        if not eff_results_list:
            return {
                'process_estimates': process_info,
                'tenpai_estimates': tenpai_probs,
                'tile_analysis': {}
            }

        for item in eff_results_list:
            if 'error' in item:
                analysis_dict[item.get('discard_tile', 'unknown')] = {
                    'error': item['error']}
                continue

            tile = item['discard_tile']
            tags = []

            live_stat = safety.get_live_info(tile)
            if live_stat:
                tags.append(live_stat)

            kabe_info = safety.get_kabe_safety_info(tile)
            if kabe_info:
                tags.extend(kabe_info)

            for pid in risky_pids:
                p_tags = []
                is_riichi = ctx['players'][pid]['riichi']
                if is_riichi:
                    if safety.check_reach_genbutsu_strict(tile, pid, pov_id):
                        p_tags.append(f"P{pid}立直现物")
                    elif normalize_tile(tile) in [normalize_tile(d) for d in ctx['players'][pid]['discards']]:
                        p_tags.append(f"P{pid}现物")
                else:
                    if normalize_tile(tile) in [normalize_tile(d) for d in ctx['players'][pid]['discards']]:
                        p_tags.append(f"P{pid}现物")

                suji_info = safety.get_suji_safety(tile, pid)
                if suji_info and is_riichi:
                    p_tags.extend([f"P{pid}{s}" for s in suji_info])

                if p_tags:
                    tags.append(" | ".join(p_tags))

            safety_str = "; ".join(tags)

            analysis_dict[tile] = {
                "shanten": item['shanten'],
                "ukeire": item['ukeire'],
                "safety_analysis": safety_str,
                "details": item.get('details', {})
            }

        return {
            "process_estimates": process_info,
            "tenpai_estimates": tenpai_probs,
            "tile_analysis": analysis_dict
        }
