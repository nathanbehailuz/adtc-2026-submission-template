#!/usr/bin/env python3
"""Merge LoRA adapters into the base HF model for GGUF conversion.

Example:
  python merge_lora.py \\
    --base Qwen/Qwen3-1.7B \\
    --adapter runs/qwen3_1_7b_qlora_v0/adapter \\
    --out runs/qwen3_1_7b_merged_v0

Run log: ``logs/merge_lora/<run>.*``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="HF base model id or path")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument(
        "--model-class",
        default="auto",
        choices=("auto", "causal", "qwen3_5"),
        help="auto tries CausalLM then Qwen3.5 multimodal class",
    )
    args = parser.parse_args()

    log = RunLogger(
        "merge_lora",
        meta={
            "base": args.base,
            "adapter": str(args.adapter),
            "out": str(args.out),
            "model_class": args.model_class,
        },
    )
    try:
        from model_loader import load_pretrained_model, load_tokenizer

        dtype = getattr(torch, args.dtype)
        log.item_start("load_base", hf_id=args.base)
        tokenizer = load_tokenizer(args.base)
        base = load_pretrained_model(
            args.base,
            model_class=args.model_class,
            torch_dtype=dtype,
            device_map="cpu",
        )
        log.item_ok("load_base", hf_id=args.base, model_class=args.model_class)

        log.item_start("load_adapter", path=str(args.adapter))
        model = PeftModel.from_pretrained(base, str(args.adapter))
        log.item_ok("load_adapter", path=str(args.adapter))

        log.item_start("merge", out=str(args.out))
        model = model.merge_and_unload()
        args.out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.out), safe_serialization=True)
        tokenizer.save_pretrained(str(args.out))
        log.item_ok("merge", out=str(args.out))
        print(f"merged model saved to {args.out}")
        print("Next: convert to GGUF with llama.cpp convert script, then quantize Q8→Q4.")
    except KeyboardInterrupt:
        log.warn("KeyboardInterrupt during merge")
        log.finish(status="interrupted", message="interrupted by user")
        raise SystemExit(130) from None
    except Exception as e:  # noqa: BLE001
        log.item_error("merge_lora", e)
        log.finish(status="error")
        raise

    summary = log.finish()
    print(f"run log -> {log.log_path}")
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
