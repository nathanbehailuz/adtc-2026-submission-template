# Merge → GGUF notes

After QLoRA SFT, adapters are not deployed directly. Pipeline:

```
adapter/  --merge_lora.py-->  qwen3_1_7b_merged_v7 (HF)
          --convert_gguf.sh-->  artifacts/gguf/adapted/qwen3_1_7b_merged_v7-{f16,Q8,Q6,Q5,Q4}_*.gguf
```

## Scripts (copies in `scripts/`)

| Script | Role |
|--------|------|
| `train_sft_qlora.py` | QLoRA SFT with TRL + PEFT |
| `merge_lora.py` | Merge LoRA into base HF weights |
| `convert_gguf.sh` | llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` (pin **b10451**) |

## Deploy artifact (Gate 2)

- Submission GGUF: **`tebeb_tutor_1.7b-Q4_K_M.gguf`**
- Source build name on train host: `qwen3_1_7b_merged_v7-Q4_K_M.gguf`
- Hosted: [`nz2212/tebebAIv2`](https://huggingface.co/nz2212/tebebAIv2) file `qwen3_1_7b_merged_v7-Q4_K_M.gguf`
- Pinned resolve URL (commit `58347614c3c7860c126f62fc7bbdb3cd1d15dd65`) in `download_model.sh`
- Local runtime path in `metadata.json`: `model/tebeb_tutor_1.7b-Q4_K_M.gguf` (download script renames into that path)

## Why Q4_K_M

EN frozen GGUF eval (Shadeform, 2026-09-22):

| Quant | AfriMGSM EN | EN STEM | Tutoring |
|-------|------------:|--------:|---------:|
| Q4_K_M | 0.428 | 0.430 | 1.0 |
| Q5_K_M | 0.432 | 0.430 | 1.0 |

Accuracy tied; Q4 preferred for laptop TPS / RSS under the ADTC scoring weights.
