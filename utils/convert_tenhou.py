import sys
import os
import re
import json
import random
import copy
import urllib.request
import gzip
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from pathlib import Path

# --- Tile conversion dictionary and functions ---

WINDS = ["東", "南", "西", "北"]
TILE_KIND_MAP = {
    "1m": 0, "2m": 1, "3m": 2, "4m": 3, "5m": 4, "0m": 4, "6m": 5, "7m": 6, "8m": 7, "9m": 8,
    "1p": 9, "2p": 10, "3p": 11, "4p": 12, "5p": 13, "0p": 13, "6p": 14, "7p": 15, "8p": 16, "9p": 17,
    "1s": 18, "2s": 19, "3s": 20, "4s": 21, "5s": 22, "0s": 22, "6s": 23, "7s": 24, "8s": 25, "9s": 26,
    "E": 27, "S": 28, "W": 29, "N": 30, "Wh": 31, "G": 32, "R": 33
}


def tile_id_to_string(tile_id):
    try:
        tile_id = int(tile_id)
    except ValueError:
        return f"?[{tile_id}]"
    if tile_id == 16: return "0m"
    if tile_id == 52: return "0p"
    if tile_id == 88: return "0s"
    suit_id = tile_id // 4
    num = (suit_id % 9) + 1
    if 0 <= suit_id <= 8:
        return f"{num}m"
    elif 9 <= suit_id <= 17:
        return f"{num}p"
    elif 18 <= suit_id <= 26:
        return f"{num}s"
    elif 27 <= suit_id <= 33:
        return ["E", "S", "W", "N", "Wh", "G", "R"][suit_id - 27]
    return f"?[{tile_id}]"


def tile_id_to_kind(tile_id):
    s = tile_id_to_string(tile_id)
    return TILE_KIND_MAP.get(s, 0)  # Default to 1m (0) to prevent errors


def get_sort_key(tile_str):
    if not tile_str or len(tile_str) < 2: return (9, 9)
    num_char = tile_str[0]
    suit_char = tile_str[-1]
    if suit_char == 'm':
        suit_ord = 0
    elif suit_char == 'p':
        suit_ord = 1
    elif suit_char == 's':
        suit_ord = 2
    else:
        suit_ord = 3
    if num_char.isdigit():
        num_ord = 5 if num_char == '0' else int(num_char)
    else:
        num_ord = "ESWNWGR".find(num_char)
    return (suit_ord, num_ord)


def format_hand_list(tile_ids):
    all_tiles_str = [tile_id_to_string(t) for t in tile_ids]
    all_tiles_str.sort(key=get_sort_key)
    return all_tiles_str


def format_meld_list(meld_list):
    return [meld['display'] for meld in meld_list]


# --- State tracking class ---

class PlayerState:
    def __init__(self, name):
        self.name = name
        self.hand = {i: [] for i in range(34)}
        self.hand_size = 0
        self.discards = []
        self.melds = []
        self.is_riichi = False
        self.score = 0

    def add_hand(self, tile_ids):
        for t_id in tile_ids:
            kind = tile_id_to_kind(t_id)
            self.hand[kind].append(t_id)
            self.hand_size += 1

    def draw(self, tile_id):
        kind = tile_id_to_kind(tile_id)
        self.hand[kind].append(tile_id)
        self.hand_size += 1

    def discard(self, tile_id):
        kind = tile_id_to_kind(tile_id)
        if tile_id in self.hand[kind]:
            self.hand[kind].remove(tile_id)
            self.hand_size -= 1
            self.discards.append(tile_id)
            return True
        elif self.hand[kind]:
            removed_tile = self.hand[kind].pop(0)
            self.hand_size -= 1
            self.discards.append(removed_tile)
            return True
        else:
            self.discards.append(tile_id)
            return False

    def remove_tiles_for_meld(self, meld_data):
        for kind in meld_data['kinds_removed_from_hand']:
            if self.hand[kind]:
                self.hand[kind].pop(0)
                self.hand_size -= 1
            else:
                # Theoretically should not happen, but just as a safeguard
                print(f"!!! Warning: P{self.name} hand is inconsistent during meld {meld_data['display']}.")

    def get_hand_tile_ids(self):
        all_ids = []
        for kind_list in self.hand.values():
            all_ids.extend(kind_list)
        return all_ids


class GameState:
    def __init__(self, init_elem, player_names):
        seed = init_elem.attrib.get('seed', '0,0,0,0,0,0').split(',')
        round_num_idx = int(seed[0])
        self.honba = int(seed[1])
        self.riichi_sticks = int(seed[2])
        self.dora_indicators = [int(seed[5])]
        self.round_wind = WINDS[round_num_idx // 4]
        self.round_deal = (round_num_idx % 4) + 1
        self.oya = int(init_elem.attrib.get('oya', 0))
        self.player_states = {}
        self.player_names = player_names  # Inject
        scores = init_elem.attrib.get('ten', '0,0,0,0').split(',')
        for i in range(4):
            name = player_names.get(i, f"Player {i}")
            p_state = PlayerState(name)
            if i in player_names:
                p_state.score = int(scores[i]) * 100
                hand_str = init_elem.attrib.get(f'hai{i}')
                if hand_str:
                    p_state.add_hand([int(t) for t in hand_str.split(',')])
            self.player_states[i] = p_state
        self.last_discard_player = -1
        self.last_discard_tile = -1
        self.last_action_was_kan = False


# --- Meld decoding ---

def decode_meld(m_code, who, last_discard_tile, is_3_player=False):
    m_code = int(m_code)
    meld = {
        "code": m_code, "from_who": -1, "type": "Unknown",
        "kinds_removed_from_hand": [], "display": ""
    }
    if is_3_player and m_code == 31520:
        meld['type'] = "KitaNuki"
        meld['from_who'] = who
        meld['kinds_removed_from_hand'] = [TILE_KIND_MAP["N"]]
        meld['display'] = "[拔北 N]"
        return meld
    from_relative = m_code & 3
    if from_relative == 0:
        meld['from_who'] = who
    else:
        meld['from_who'] = (who - from_relative + 4) % 4
    if m_code & (1 << 2):
        meld['type'] = "Chii"
        t0 = (m_code >> 10) // 3
        base = (t0 // 7) * 9 + (t0 % 7)
        kind0 = tile_id_to_kind(base * 4)
        kind1 = tile_id_to_kind((base + 1) * 4)
        kind2 = tile_id_to_kind((base + 2) * 4)
        called_kind = tile_id_to_kind(last_discard_tile)
        if kind0 == called_kind:
            meld['kinds_removed_from_hand'] = [kind1, kind2]
        elif kind1 == called_kind:
            meld['kinds_removed_from_hand'] = [kind0, kind2]
        else:
            meld['kinds_removed_from_hand'] = [kind0, kind1]
        meld['display'] = f"[Chii {tile_id_to_string(base * 4)} {tile_id_to_string((base + 1) * 4)} {tile_id_to_string((base + 2) * 4)} from P{meld['from_who']}]"
    elif m_code & (1 << 3):
        meld['type'] = "Pon"
        t_base = (m_code >> 9) // 3
        kind = tile_id_to_kind(t_base * 4)
        meld['kinds_removed_from_hand'] = [kind, kind]
        tile_str = tile_id_to_string(t_base * 4)
        meld['display'] = f"[Pon {tile_str} {tile_str} {tile_str} from P{meld['from_who']}]"
    elif m_code & (1 << 4):
        meld['type'] = "Kakan"
        t_base = (m_code >> 9) // 3
        kind = tile_id_to_kind(t_base * 4)
        meld['kinds_removed_from_hand'] = [kind]
        tile_str = tile_id_to_string(t_base * 4)
        meld['display'] = f"[Kakan {tile_str} {tile_str} {tile_str} {tile_str} from P{meld['from_who']}]"
    else:
        t_base = (m_code >> 8) // 4
        kind = tile_id_to_kind(t_base * 4)
        tile_str = tile_id_to_string(t_base * 4)
        if from_relative == 0:
            meld['type'] = "Ankan"
            meld['kinds_removed_from_hand'] = [kind, kind, kind, kind]
            meld['display'] = f"[Ankan {tile_str} {tile_str} {tile_str} {tile_str} from P{meld['from_who']}]"
        else:
            meld['type'] = "Daiminkan"
            meld['kinds_removed_from_hand'] = [kind, kind, kind]
            meld['display'] = f"[Daiminkan {tile_str} {tile_str} {tile_str} {tile_str} from P{meld['from_who']}]"
    return meld


# --- Core public functions ---

def read_and_clean_xml_from_string(content):
    """Read and clean XML from string content"""
    try:
        xml_start = content.find("<mjloggm")
        xml_end = content.find("</mjloggm>")
        if xml_start == -1: raise ValueError("Tag <mjloggm> not found.")

        if xml_end == -1:
            clean_xml = content[xml_start:] + "</mjloggm>"
        else:
            clean_xml = content[xml_start:xml_end + 10]

        if not clean_xml.strip().endswith("</mjloggm>"):
            last_lt = clean_xml.rfind('<')
            if last_lt > clean_xml.rfind('>'): clean_xml = clean_xml[:last_lt]
            clean_xml += "</mjloggm>"

        return ET.fromstring(clean_xml)
    except Exception as e:
        print(f"Error: Failed to parse XML. {e}")
        return None


def _parse_owari_scores(owari_attr, player_names):
    """Helper function: parse owari tag"""
    scores = owari_attr.split(',')
    ranking = []
    num_players = len(player_names)

    for i in range(num_players):
        if i in player_names:
            final_score_uma = float(scores[i * 2 + 1])
            raw_score = int(scores[i * 2]) * 100
            ranking.append({
                "id": i,
                "name": player_names[i],
                "final_score_uma": final_score_uma,
                "raw_score": raw_score
            })

    ranking.sort(key=lambda x: x['final_score_uma'], reverse=True)
    return ranking


def analyze_game_outcomes(root):
    """
    Analyze game outcomes from XML root element.
    """
    if root is None:
        return None

    player_names = {}
    player_ranks = {}  # Add a dictionary to store ranks
    final_ranking = []
    ron_events_by_round_idx = {}
    round_idx = -1
    is_3_player = False

    for elem in root:
        tag = elem.tag

        if tag == 'GO':
            game_type = int(elem.attrib.get('type', 0))
            is_3_player = (game_type & 0x10) != 0

        elif tag == 'UN':
            num_players = 3 if is_3_player else 4
            dan_list_str = elem.attrib.get('dan', '').split(',')  # Get rank list

            for i in range(num_players):
                key = f'n{i}'
                if key in elem.attrib and elem.attrib[key]:
                    player_names[i] = unquote(elem.attrib[key])
                    try:
                        # Store player ranks, e.g., "16" -> 16
                        player_ranks[i] = int(dan_list_str[i])
                    except (IndexError, ValueError):
                        # If rank info is missing or invalid, record as 0
                        player_ranks[i] = 0

        elif tag == 'INIT':
            round_idx += 1

        elif tag == 'AGARI':
            who = int(elem.attrib.get('who'))
            fromWho = int(elem.attrib.get('fromWho'))

            # Use str(round_idx) as key
            if who == fromWho:
                ron_events_by_round_idx[str(round_idx)] = {"winner_id": who, "loser_id": None}
            else:
                ron_events_by_round_idx[str(round_idx)] = {"winner_id": who, "loser_id": fromWho}

            owari = elem.attrib.get('owari')
            if owari and not final_ranking:
                final_ranking = _parse_owari_scores(owari, player_names)

        elif tag == 'RYUUKYOKU':
            ron_events_by_round_idx[str(round_idx)] = None  # Ryuukyoku (drawn game)

            owari = elem.attrib.get('owari')
            if owari and not final_ranking:
                final_ranking = _parse_owari_scores(owari, player_names)

    return {
        "player_names": player_names,
        "player_ranks": player_ranks,  # Include rank info in the returned results
        "final_ranking": final_ranking,
        "ron_events_by_round_idx": ron_events_by_round_idx
    }


def create_snapshot(gamestate, pov_player_id, drawn_tile_id):
    """
    Create a JSON snapshot of the game state.
    """
    pov_state = gamestate.player_states[pov_player_id]
    current_hand_ids = pov_state.get_hand_tile_ids() + [drawn_tile_id]

    state_json = {
        "round_info": {
            "round": f"{gamestate.round_wind}{gamestate.round_deal}局",
            "honba": gamestate.honba,
            "riichi_sticks": gamestate.riichi_sticks,
            "oya_player": gamestate.oya
        },
        "dora_indicators": [tile_id_to_string(t) for t in gamestate.dora_indicators],
        "pov_player": {
            "player_id": pov_player_id,
            "score": pov_state.score,
            "hand": format_hand_list(current_hand_ids),
            "drawn_tile": tile_id_to_string(drawn_tile_id),
            "is_riichi": pov_state.is_riichi
        },
        "public_info": {}
    }

    for i, p_state in gamestate.player_states.items():
        if i in gamestate.player_names:
            state_json["public_info"][f"player_{i}"] = {
                "name": p_state.name,
                "score": p_state.score,
                "discards": format_hand_list(p_state.discards),
                "melds": format_meld_list(p_state.melds),
                "is_riichi": p_state.is_riichi
            }
    return state_json


def generate_slices_for_pov(root, pov_player_id):
    """
    Generate all slices for a *single* player from the XML root element.
    """
    if root is None:
        return []

    dataset = []
    gamestate = None
    player_names = {}
    pending_snapshot = None
    is_3_player = False
    draw_map = {'T': 0, 'U': 1, 'V': 2, 'W': 3}
    discard_map = {'D': 0, 'E': 1, 'F': 2, 'G': 3}

    for elem in root:
        tag = elem.tag

        if tag == 'GO':
            game_type = int(elem.attrib.get('type', 0))
            is_3_player = (game_type & 0x10) != 0

        elif tag == 'UN':
            num_players = 3 if is_3_player else 4
            for i in range(num_players):
                key = f'n{i}'
                if key in elem.attrib and elem.attrib[key]:
                    player_names[i] = unquote(elem.attrib[key])

        elif tag == 'INIT':
            pending_snapshot = None
            gamestate = GameState(elem, player_names)

        elif gamestate:
            if tag == 'N':
                if pending_snapshot:
                    pending_snapshot = None
                who = int(elem.attrib.get('who'))
                m_code = int(elem.attrib.get('m'))
                meld_data = decode_meld(m_code, who, gamestate.last_discard_tile, is_3_player)
                gamestate.player_states[who].melds.append(meld_data)
                gamestate.player_states[who].remove_tiles_for_meld(meld_data)
                if meld_data['type'] in ['Ankan', 'Kakan', 'Daiminkan', 'KitaNuki']:
                    gamestate.last_action_was_kan = True

            elif tag == 'REACH':
                who = int(elem.attrib.get('who'))
                step = int(elem.attrib.get('step'))
                if step == 1:
                    gamestate.player_states[who].is_riichi = True
                elif step == 2:
                    gamestate.player_states[who].score -= 1000

            elif tag == 'DORA':
                gamestate.dora_indicators.append(int(elem.attrib.get('hai')))

            elif tag == 'AGARI' or tag == 'RYUUKYOKU':
                pending_snapshot = None
                gamestate = None

            elif tag[0] in draw_map:
                player_id = draw_map[tag[0]]
                if player_id not in player_names: continue

                tile_id = int(tag[1:])
                if gamestate.last_action_was_kan:
                    gamestate.player_states[player_id].draw(tile_id)
                    gamestate.last_action_was_kan = False
                else:
                    if player_id == pov_player_id:
                        if pending_snapshot:
                            print(f"Warning: P{pov_player_id} drew consecutively without discarding, discarding previous snapshot.")
                        state_json = create_snapshot(gamestate, pov_player_id, tile_id)
                        pending_snapshot = {"state": state_json, "action": None}
                    gamestate.player_states[player_id].draw(tile_id)

            elif tag[0] in discard_map:
                player_id = discard_map[tag[0]]
                if player_id not in player_names: continue

                tile_id = int(tag[1:])
                gamestate.player_states[player_id].discard(tile_id)
                gamestate.last_discard_player = player_id
                gamestate.last_discard_tile = tile_id

                if pending_snapshot and player_id == pov_player_id:
                    pending_snapshot["action"] = tile_id_to_string(tile_id)
                    dataset.append(pending_snapshot)
                    pending_snapshot = None

    return dataset
HEADER = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive'
}


def download_log_content(log_url):
    """
    Download and decompress log XML content from Tenhou server.
    """
    try:
        if "?log=" not in log_url:
            print(f"  [!] Invalid URL format: {log_url}")
            return None

        data_url = log_url.replace("?log=", "log/?")

        domain = data_url.split('/')[2]  # e.g., tenhou.net
        req_header = HEADER.copy()
        req_header['Host'] = domain

        req = urllib.request.Request(url=data_url, headers=req_header)
        opener = urllib.request.build_opener()

        with opener.open(req, timeout=15) as response:
            content = response.read()

            if response.info().get('Content-Encoding') == 'gzip':
                content = gzip.decompress(content)

            return content.decode('utf-8')

    except Exception as e:
        print(f"  [!] Download failed {log_url}: {e}")
        return None


def process_single_log(log_xml_content, log_id):
    """
    Helper function encapsulating the full slicing logic.
    Processes everything in memory and returns the final slice list.
    """
    try:
        # 1. Parse XML from string
        root = read_and_clean_xml_from_string(log_xml_content)
        if root is None:
            print(f"  [!] {log_id}: XML parsing failed.")
            return None

        # 2. Analyze outcomes (in memory)
        outcomes = analyze_game_outcomes(root)
        if not outcomes or not outcomes['final_ranking']:
            print(f"  [!] {log_id}: Failed to analyze game outcomes.")
            return None

        # ----------------------------------------------------
        # Check ranks
        player_ranks = outcomes.get('player_ranks', {})
        if not player_ranks:
            print(f"  [!] {log_id}: Failed to parse player ranks. Skipping.")
            return None

        # Check if any player rank is below 16 (7th Dan)
        min_rank = 99
        if player_ranks:  # Ensure dictionary is not empty
            min_rank = min(player_ranks.values())

        if min_rank < 16:
            print(f"  [!] {log_id}: Filtering failed, contains low rank player (Lowest: {min_rank} Dan). Skipping.")
            return None
        # ----------------------------------------------------

        # 3. Determine P1 and P2
        p1 = outcomes['final_ranking'][0]
        p2 = outcomes['final_ranking'][1]
        pov_choices = [p1['id'], p2['id']]
        weights = [p1['raw_score'] if p1['raw_score'] > 0 else 1,
                   p2['raw_score'] if p2['raw_score'] > 0 else 1]

        # 4. Generate all slices for P1 and P2
        p1_all_slices = generate_slices_for_pov(root, p1['id'])
        p2_all_slices = generate_slices_for_pov(root, p2['id'])

        # 5. Filter by round
        final_training_set = []
        ron_losers_by_round_name = {}

        round_idx = -1
        # Iterate through XML to find rounds
        for elem in root:
            if elem.tag == 'INIT':
                round_idx += 1
                seed = elem.attrib.get('seed', '0,0,0,0,0,0').split(',')
                round_num_idx = int(seed[0])
                honba = int(seed[1])
                round_wind = WINDS[round_num_idx // 4]  # WINDS is now imported
                round_deal = (round_num_idx % 4) + 1
                round_name = f"{round_wind}{round_deal}局 {honba}本场"

                # Get the player who dealt in this round
                ron_event = outcomes['ron_events_by_round_idx'].get(str(round_idx))
                loser_id = None
                if ron_event and ron_event['loser_id'] is not None:
                    loser_id = ron_event['loser_id']

                # Determine POV for this round
                chosen_pov_id = random.choices(pov_choices, weights=weights, k=1)[0]
                if loser_id == chosen_pov_id:
                    chosen_pov_id = p1['id'] if chosen_pov_id == p2['id'] else p2['id']

                # Filter slices belonging to this round and the chosen POV
                for sl in p1_all_slices:
                    sl_round_name = sl['state']['round_info']['round'] + f" {sl['state']['round_info']['honba']}本场"
                    if sl_round_name == round_name and sl['state']['pov_player']['player_id'] == chosen_pov_id:
                        final_training_set.append(sl)

                for sl in p2_all_slices:
                    sl_round_name = sl['state']['round_info']['round'] + f" {sl['state']['round_info']['honba']}本场"
                    if sl_round_name == round_name and sl['state']['pov_player']['player_id'] == chosen_pov_id:
                        final_training_set.append(sl)

        return final_training_set

    except Exception as e:
        print(f"  [!] Unexpected error processing {log_id}: {e}")
        return None


def main_batch_processor(html_filename):
    """
    Main batch processing function
    """
    # Target room name to filter
    target_room_name = "四鳳南喰赤－"

    # 1. Create output directory
    output_dir = Path(Path(html_filename).stem)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: ./{output_dir.name}/")

    # 2. Compile regex to extract URLs
    url_regex = re.compile(r'<a href="(http://tenhou.net/0/\?log=[^"]+)">')

    processed_count = 0
    total_slices_generated = 0

    try:
        print(f"Opening {html_filename}...")
        # Ensure HTML file is opened with utf-8 encoding
        with open(html_filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # 3. Filter room name
                if target_room_name not in line:
                    continue

                # 4. Extract URL
                match = url_regex.search(line)
                if not match:
                    continue

                log_url = match.group(1)
                log_id = log_url.split('=')[-1]
                print(f"\n[Game {processed_count + 1}] Found {target_room_name} log: {log_id}")

                # 5. Download log content
                print(f"  Downloading...")
                log_xml_content = download_log_content(log_url)
                if not log_xml_content:
                    print(f"  -> Download failed, skipping.")
                    continue

                # 6. Process log in memory and apply filtering logic
                print(f"  Processing and filtering...")
                final_training_set = process_single_log(log_xml_content, log_id)

                # 7. Save
                if final_training_set:
                    processed_count += 1
                    output_filename = output_dir / f"{processed_count}.json"
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(final_training_set, f, indent=2, ensure_ascii=False)
                    print(f"  -> Success: Saved {len(final_training_set)} slices to {output_filename}")
                    total_slices_generated += len(final_training_set)
                else:
                    print(f"  -> No valid slices generated for P1/P2, skipping.")

    except FileNotFoundError:
        print(f"Error: HTML file '{html_filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Unknown error occurred: {e}")
        sys.exit(1)

    print(f"\n--- Batch processing complete ---")
    print(f"Total processed and saved logs: {processed_count}.")
    print(f"Total training slices generated: {total_slices_generated}.")
    print(f"All files saved in ./{output_dir.name}/ directory.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_processor.py <html_index_filename>")
        print("Example: python batch_processor.py scc20241231.html")
        sys.exit(1)

    html_filename_arg = sys.argv[1]
    main_batch_processor(html_filename_arg)
