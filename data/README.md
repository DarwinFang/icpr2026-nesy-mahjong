# Dataset files

Download the dataset from [DarwinFang/nesy-mahjong](https://huggingface.co/datasets/DarwinFang/nesy-mahjong) and place the files in this directory.

## Main splits

| File | Samples | Description |
|---|---:|---|
| `train.jsonl` | 44,034 | Stage 1 decision training |
| `validation.jsonl` | 4,416 | Stage 1 validation |
| `test.jsonl` | 4,896 | Chronologically held-out test set |
| `stage2_reasoning.jsonl` | 2,500 | Public teacher-distilled reasoning subset |

The paper reports results using 5,544 Stage 2 training examples. The currently published `stage2_reasoning.jsonl` is a 2,500-example subset.

## Stratified test files

Place the following files in `data/split_test/`:

| File | Samples | Context |
|---|---:|---|
| `test_early_offense.jsonl` | 816 | Early, no opponent Riichi |
| `test_early_defense.jsonl` | 816 | Early, at least one opponent Riichi |
| `test_mid_offense.jsonl` | 816 | Mid, no opponent Riichi |
| `test_mid_defense.jsonl` | 816 | Mid, at least one opponent Riichi |
| `test_late_offense.jsonl` | 816 | Late, no opponent Riichi |
| `test_late_defense.jsonl` | 816 | Late, at least one opponent Riichi |
| `test_combined.jsonl` | 4,896 | All six Tenhou test buckets |
| `majsoul_challenge_20.jsonl` | 20 | Mahjong Soul human-alignment benchmark |

Game phase is determined by the ego-player's discard count: early is 0-5, mid is 6-11, and late is 12 or more. Defense means that at least one opponent has declared Riichi.
