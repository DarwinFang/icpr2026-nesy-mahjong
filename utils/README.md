# Tenhou Log Converter

Extracts Riichi Mahjong game logs from Tenhou HTML archives into JSON slices for training.

## Quick Start

```bash
python utils/convert_tenhou.py scc20241231.html
```

This creates a `scc20241231/` directory containing one JSON file per qualifying game.

## How It Works

1. **Download the archive** — Go to https://tenhou.net/sc/raw/, pick a year under "過去ログ", download and extract. Inside you'll find `sccYYYYMMDD.html` files (one per day).
2. **Run the converter** — The script reads the HTML index, downloads each game log from Tenhou's servers, parses the XML into structured JSON, and generates training slices for the top two finishers.
3. **Only high-rank games** — By default, games where any player is below 7-dan are skipped.

## Configuration

All tunable parameters are near the top of `convert_tenhou.py`:

| Parameter | Line | Default | Description |
|-----------|------|---------|-------------|
| `target_room_name` | ~602 | `"四鳳南喰赤－"` | Tenhou room name to match. Change to `"四鳳南喰赤"` (no `－`) or `"四特南喰赤－"` for different lobbies. |
| `min_rank < 16` | ~536 | `16` (7-dan) | Minimum player rank filter. `16` = 7-dan, `20` = 10-dan. Set to `0` to accept all ranks. |
| `HEADER` | ~466 | Firefox UA | HTTP request headers used when downloading logs from Tenhou. |

## Output Format

Each JSON file contains an array of game slices:

```json
[
  {
    "state": {
      "round_info": {"round": "東1局", "honba": 0, "riichi_sticks": 0, "oya_player": 0},
      "dora_indicators": ["9m"],
      "pov_player": {
        "player_id": 0,
        "score": 25000,
        "hand": ["2m", "2m", "3p", ...],
        "drawn_tile": "9p",
        "is_riichi": false
      },
      "public_info": {
        "player_0": {"name": "...", "score": 25000, "discards": [...], "melds": [...], "is_riichi": false},
        ...
      }
    },
    "action": "9p"
  },
  ...
]
```

## Reference

This script is based on the Tenhou log parsing approach described in:
https://notoootori.github.io/2020/07/28/%E5%A4%A9%E5%87%A4%E7%89%8C%E8%B0%B1%E9%87%87%E9%9B%86%E5%8F%8A%E5%88%86%E6%9E%90.html (Chinese)

## License

MIT
