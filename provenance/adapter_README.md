# TebebAI LoRA adapter (QLoRA v7)

| Field | Value |
|-------|--------|
| Base | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Method | QLoRA → PEFT LoRA adapters |
| Files | `adapter_model.safetensors`, `adapter_config.json` |
| Config | `qlora_qwen3_1_7b_v7.yaml` (r=32, α=64, 2 epochs) |
| Train script | `scripts/train_sft_qlora.py` |

These adapters are proof-of-training for Gate 2. The **deployed** artifact is the merged+quantized GGUF downloaded by `download_model.sh`, not this folder.
