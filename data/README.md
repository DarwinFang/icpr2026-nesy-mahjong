# Datasets

Download from [HuggingFace Datasets](https://huggingface.co/datasets/DarwinFang/nesy-mahjong) and place files here.

## Files

| File | Size | Description |
|------|------|-------------|
| train.jsonl | ~32 MB | Stage 1: game state -> discard decision |
| test.jsonl | ~4 MB | Test set |
| validation.jsonl | ~3 MB | Validation set |
| stage2_reasoning.jsonl | ~4 MB | Stage 2: teacher-distilled reasoning data |

## Split test sets

Place in :

| File | Description |
|------|-------------|
| test_early_offense.jsonl | Early round, offensive |
| test_early_defense.jsonl | Early round, defensive |
| test_mid_offense.jsonl | Mid round, offensive |
| test_mid_defense.jsonl | Mid round, defensive |
| test_late_offense.jsonl | Late round, offensive |
| test_late_defense.jsonl | Late round, defensive |
| test_combined.jsonl | All combined |
| majsoul_challenge_20.jsonl | Majsoul challenge |
