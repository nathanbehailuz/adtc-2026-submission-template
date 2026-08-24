#!/usr/bin/env python3
"""Terminal chat with the TebebAI STEM tutor GGUF (offline, llama.cpp).

  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  bash download_model.sh
  python chat.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "tebeb_tutor_1.7b.gguf"
METADATA = ROOT / "metadata.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SYSTEM_PROMPT = """You are TebebAI, an English STEM tutor that runs fully offline.

- Explain clearly for a student: solve step-by-step, give hints without revealing final answers, or diagnose first errors when asked.
- Keep equations, expressions, numbers, variables, operators, and fractions in standard mathematical notation.
- Be concise and helpful."""


def resolve_model_path(cli: Path | None) -> Path:
    if cli is not None:
        return cli.expanduser().resolve()
    if METADATA.is_file():
        try:
            data = json.loads(METADATA.read_text(encoding="utf-8"))
            rel = (data.get("_runtime") or {}).get("model_path")
            if rel:
                return (ROOT / rel).resolve()
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_MODEL.resolve()


def clean_response(text: str) -> str:
    """Strip Qwen think tags and GSM8K <<calc>> spans for display."""
    s = text or ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in s:
        s = s.split("</think>")[-1]
    s = re.sub(r"</?think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<<.*?>>", "", s, flags=re.DOTALL)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def default_n_threads() -> int:
    """Use all CPUs available to this process (cgroup/affinity-aware on Linux)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, int(n))


def apply_thread_caps(n: int) -> None:
    n_s = str(max(1, n))
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = n_s


def check_model_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(
            f"[error] Model not found: {path}\n"
            "  Run:  bash download_model.sh\n"
            "  Or pass:  python chat.py --model /path/to/model.gguf"
        )
    if path.stat().st_size == 0:
        raise SystemExit(f"[error] Model file is empty: {path}")


def load_llm(path: Path, n_ctx: int, n_threads: int):
    apply_thread_caps(n_threads)
    print("Importing llama_cpp …", flush=True)
    t0 = time.perf_counter()
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise SystemExit(
            "[error] llama_cpp not installed.\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt"
        ) from e
    print(f"Imported in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"Loading {path.name}  (threads={n_threads}, n_ctx={n_ctx}) …", flush=True)
    t1 = time.perf_counter()
    try:
        llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[error] Failed to load model: {e}") from e
    print(f"Loaded in {time.perf_counter() - t1:.1f}s\n", flush=True)
    return llm


def print_intro(model_path: Path) -> None:
    bar = "=" * 60
    print(bar)
    print("  TebebAI — Offline English STEM Tutor")
    print("  Qwen3-1.7B · llama.cpp · Q5_K_M")
    print(bar)
    print(f"  Model: {model_path}")
    print()
    print("  Ask math or science questions. I can solve, explain,")
    print("  hint (without giving the answer), or find first errors.")
    print()
    print("  Commands:  quit / exit / q   |   clear / reset")
    print(bar)
    print()


def user_content(text: str) -> str:
    if text.lstrip().startswith("/no_think"):
        return text
    return f"/no_think\n{text}"


def chat_loop(llm, max_tokens: int, temperature: float) -> None:
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            raw = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not raw:
            continue
        low = raw.lower()
        if low in {"q", "quit", "exit"}:
            print("Bye.")
            break
        if low in {"clear", "reset"}:
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(conversation cleared)\n")
            continue

        history.append({"role": "user", "content": user_content(raw)})
        print("TebebAI> ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            out = llm.create_chat_completion(
                messages=history,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            reply = out["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            history.pop()
            print(f"\n[error] generation failed: {e}\n")
            continue
        cleaned = clean_response(reply)
        elapsed = time.perf_counter() - t0
        print(f"{cleaned}\n  ({elapsed:.1f}s)\n")
        history.append({"role": "assistant", "content": cleaned})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None, help="Path to .gguf (default: metadata _runtime.model_path)")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)
    check_model_file(model_path)
    n_threads = args.n_threads if args.n_threads is not None else default_n_threads()
    llm = load_llm(model_path, n_ctx=args.n_ctx, n_threads=n_threads)
    print_intro(model_path)
    chat_loop(llm, max_tokens=args.max_tokens, temperature=args.temperature)


if __name__ == "__main__":
    main()
