# Dataset info — TebebAI SFT mix v7

## Summary

English-only supervised fine-tuning mix used for QLoRA v7 on `Qwen/Qwen3-1.7B`.

| Source | Rows | Notes |
|--------|-----:|-------|
| `openai/gsm8k` (train) | 7473 | Cleaned: stripped `####` / `<<expr=val>>` markup; templated into solve / explain / hint / first_error |
| `allenai/sciq` (train subset) | 3000 | Science Q&A rewritten into tutoring behaviors |
| `authored_tutoring_v7` | 163 | Hand-authored multi-constraint STEM tutoring (judge-aligned) |
| **Total** | **10636** | `data/train/sft_mix_v7.jsonl` |

Builder: `adtc/data/mix_sft_v7.py` (see also `adtc/data/build_authored_tutoring_v7.py`).

## Checksums

| Artifact | SHA256 |
|----------|--------|
| Full mix `sft_mix_v7.jsonl` (not committed here) | `605184a9505e83b7893ae1205b985c49fbc6c2ddad00819b8fdbb18d395c3b04` |
| Sample (3 rows) `samples/sft_mix_v7_sample.jsonl` | see `checksums.sha256` |
| Adapter `adapter_model.safetensors` | see `checksums.sha256` |

## Row schema

Each JSONL line:

```json
{
  "id": "...",
  "direction": "en_en",
  "behavior": "solve|explain|hint|first_error",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "gsm8k_train_v7|sciq_train_v7|authored_tutoring_v7"
}
```

## Behavior mix (approx.)

| Behavior | Count |
|----------|------:|
| hint | 2937 |
| solve | 2903 |
| explain | 2901 |
| first_error | 1895 |

## Dedup / eval hygiene

Training mix is deduped against frozen EN eval holdouts (`en_stem_holdout_v0`, `afrimgsm_eng_test_v0`) during mix build.

## Sample excerpt

See `samples/sft_mix_v7_sample.jsonl` (one hint, one first_error, one explain row).
