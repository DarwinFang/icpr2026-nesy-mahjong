"""Shared data handling and prompt formatting for benchmark scripts."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_TILES = {
    "1m", "2m", "3m", "4m", "5m", "0m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "0p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "0s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "Wh", "G", "R",
}

TEST_BUCKETS = (
    "test_early_offense.jsonl",
    "test_mid_offense.jsonl",
    "test_late_offense.jsonl",
    "test_early_defense.jsonl",
    "test_mid_defense.jsonl",
    "test_late_defense.jsonl",
)

TILE_TOKEN = r"(?:[0-9][mps]|Wh|[ESWNGR])"
TILE_SEARCH = re.compile(rf"(?<![A-Za-z0-9])({TILE_TOKEN})(?![A-Za-z0-9])")
DOLLAR_TILE_SEARCH = re.compile(rf"\$\s*({TILE_TOKEN})")
JSON_TILE_SEARCH = re.compile(rf'"discard_tile"\s*:\s*"({TILE_TOKEN})"')


def parse_bool(value: Any) -> bool:
    """Parse a command-line boolean accepted by Fire and argparse."""
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def clean_tile_str(tile: Any) -> str:
    """Normalize an evaluation label without changing red-five notation."""
    if not isinstance(tile, str):
        return ""
    return tile.replace("*", "").strip()


def get_ground_truth_discard(output_field: Any) -> str:
    """Read a discard label stored as a dict, JSON string, or bare tile."""
    value = output_field
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return clean_tile_str(value)
    if isinstance(value, dict):
        return clean_tile_str(value.get("discard_tile"))
    return ""


def extract_generated_discard(text: str, prefer_dollar: bool = False) -> str:
    """Extract a discard from a model response without confusing ``Wh`` with ``W``."""
    if not isinstance(text, str):
        return ""
    response = text.strip()
    if response in VALID_TILES:
        return response

    matchers = (
        (DOLLAR_TILE_SEARCH, JSON_TILE_SEARCH)
        if prefer_dollar
        else (JSON_TILE_SEARCH, DOLLAR_TILE_SEARCH)
    )
    for pattern in matchers:
        match = pattern.search(response)
        if match:
            return match.group(1)

    match = TILE_SEARCH.search(response)
    return match.group(1) if match else ""


def extract_thoughts(text: str) -> str:
    """Extract the ``thoughts`` field when generation returns JSON."""
    if not isinstance(text, str):
        return ""
    response = text.strip()
    decoder = json.JSONDecoder()
    object_start = response.find("{")
    if object_start >= 0:
        try:
            value, _ = decoder.raw_decode(response[object_start:])
            if isinstance(value, dict) and isinstance(value.get("thoughts"), str):
                return value["thoughts"]
        except json.JSONDecodeError:
            pass
    return response


def format_comparative_data(analysis_result: Dict[str, Any], target_tile: str) -> str:
    """Format calculator output exactly as used for Stage 2 training."""
    if not analysis_result or "tile_analysis" not in analysis_result:
        return "（客观数据计算失败）"

    analysis = analysis_result["tile_analysis"]
    tenpai = ", ".join(
        f"P{player_id}: {status}"
        for player_id, status in analysis_result.get("tenpai_estimates", {}).items()
    )
    phase = analysis_result.get("process_estimates", "未知巡目")

    options = []
    selected = None
    for tile, data in analysis.items():
        if "error" in data:
            continue
        option = {
            "tile": tile,
            "shanten": data.get("shanten", 99),
            "ukeire": data.get("ukeire", 0),
            "safety": data.get("safety_analysis") or "-",
        }
        options.append(option)
        if tile == target_tile:
            selected = option

    if selected is None:
        selected = {
            "tile": target_tile,
            "shanten": 99,
            "ukeire": 0,
            "safety": "未知",
        }
        options.append(selected)

    options.sort(key=lambda item: (item["shanten"], -item["ukeire"]))
    displayed = options[:4]
    if selected not in displayed:
        displayed.append(selected)

    lines = [
        f"全局形势：【{phase}】听牌概率：{tenpai}",
        "【切牌选项对比表】（按牌效从高到低排序）：",
    ]
    for option in displayed:
        marker = " 👈(实战决策)" if option["tile"] == target_tile else ""
        lines.append(
            f"- [切{option['tile']}]: {option['shanten']}向听, "
            f"进{option['ukeire']}张 | {option['safety']}{marker}"
        )
    return "\n".join(lines)


def load_jsonl(filename: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a JSONL file, reporting malformed records instead of hiding them."""
    path = Path(filename)
    if not path.exists():
        return []

    samples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(samples) >= limit:
                break
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}") from exc
    return samples


def save_json(data: Any, filename: str) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
