#!/usr/bin/env python3
"""Offline TebebAI STEM tutor chat.

Reads the GGUF path from metadata.json `_runtime.model_path`.
Multi-turn history. Strips <think> blocks and GSM8K <<>> / #### markup.

Backends (first available):
  1. llama-cpp-python
  2. llama-cli (pinned llama.cpp b10451 under adtc/tools/)

  python chat.py
  python chat.py --from-metadata
  python chat.py --prompt "Give one hint for 2x+7=19 without revealing x."
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "metadata.json"

SYSTEM_PROMPT = """You are an English STEM tutor for secondary-school students.

Behavior:
- When asked to solve: give clear numbered steps with brief justifications; end with "Final answer: …".
- When asked for a hint: give one next step, explain why it is valid, and ask one short check question. Do not reveal the final answer.
- When asked to find a student's mistake: name the exact wrong step, explain why it is wrong, give a corrective hint, and ask one follow-up question. Do not reveal the final answer.
- When the user lists multiple requirements, address every one (checklist). Prefer concise, complete replies over vague length.
- Keep equations and numbers in standard math notation.

Never use GSM8K markup such as #### or <<expr=val>> in your replies."""

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
MARKUP_CALC_RE = re.compile(r"<<[^>]*>>")
MARKUP_HASH_RE = re.compile(r"####\s*\S+")


def load_meta() -> dict:
    if not META_PATH.is_file():
        raise SystemExit(f"missing {META_PATH}")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def resolve_gguf(meta: dict) -> Path:
    rel = (meta.get("_runtime") or {}).get("model_path")
    if not rel:
        raise SystemExit("metadata.json is missing _runtime.model_path")
    path = (ROOT / rel).resolve()
    if not path.is_file():
        raise SystemExit(
            f"GGUF not found: {path}\n"
            "Run: bash download_model.sh\n"
            "Or copy/symlink the local Q4 file to that path."
        )
    return path


def clean_reply(text: str) -> str:
    text = text or ""
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = MARKUP_CALC_RE.sub("", text)
    text = MARKUP_HASH_RE.sub("", text)
    return text.strip()


def user_turn(text: str) -> str:
    text = text.strip()
    if text.lstrip().startswith("/no_think"):
        return text
    return f"/no_think\n{text}"


def find_llama_cli() -> Path | None:
    env = os.environ.get("LLAMA_CLI")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    which = shutil.which("llama-cli")
    if which:
        return Path(which)
    parent = ROOT.parent
    for cand in (
        parent / "adtc/tools/llama.cpp/llama-b10451/llama-cli",
        parent / "adtc/tools/llama.cpp/b10451/llama-b10451/llama-cli",
    ):
        if cand.is_file():
            return cand
    return None


def format_chatml(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _strip_llama_cli_banner(text: str) -> str:
    """Keep the assistant reply from llama-cli --single-turn output."""
    if "\n> " in text:
        text = text.split("\n> ", 1)[-1]
    lines = text.splitlines()
    while lines and (lines[0].startswith(">") or lines[0].startswith("/no_think") or not lines[0].strip()):
        lines.pop(0)
    if lines:
        lines.pop(0)  # echoed user prompt
    cut = []
    for line in lines:
        if line.startswith("[ Prompt:") or line.startswith("Exiting"):
            break
        cut.append(line)
    return clean_reply("\n".join(cut))


class LlamaCppBackend:
    def __init__(self, path: Path, n_ctx: int, n_threads: int, n_gpu_layers: int):
        from llama_cpp import Llama

        print(f"Loading {path.name} via llama-cpp-python (ctx={n_ctx}, threads={n_threads}, ngl={n_gpu_layers}) …", flush=True)
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print(f"Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return clean_reply(out["choices"][0]["message"]["content"] or "")


class LlamaCliBackend:
    def __init__(self, path: Path, cli: Path, n_ctx: int, n_threads: int, n_gpu_layers: int):
        self.path = path
        self.cli = cli
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        print(f"Using llama-cli {cli}  model={path.name}", flush=True)

    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
        user_parts = [m for m in messages if m["role"] != "system"]
        if len(user_parts) == 1 and user_parts[0]["role"] == "user":
            user_text = user_parts[0]["content"]
        else:
            user_text = format_chatml(user_parts)
        sys_file = usr_file = out_file = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sys.txt", delete=False) as sf:
                sf.write(system)
                sys_file = sf.name
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".usr.txt", delete=False) as uf:
                uf.write(user_text)
                usr_file = uf.name
            with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".out.txt", delete=False) as of:
                out_file = of.name
            cmd = [
                str(self.cli),
                "-m", str(self.path),
                "-sysf", sys_file,
                "-f", usr_file,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "-c", str(self.n_ctx),
                "-t", str(self.n_threads),
                "-ngl", str(self.n_gpu_layers if self.n_gpu_layers >= 0 else 99),
                "--single-turn",
                "--jinja",
                "--reasoning", "off",
                "--log-disable",
                "--simple-io",
            ]
            with open(out_file, "w", encoding="utf-8") as out_fh:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=out_fh,
                    stderr=out_fh,
                )
            text = Path(out_file).read_text(encoding="utf-8", errors="replace")
        finally:
            for p in (sys_file, usr_file, out_file):
                if p:
                    Path(p).unlink(missing_ok=True)
        if proc.returncode != 0:
            raise SystemExit(f"llama-cli failed (exit {proc.returncode}):\n{text[-2000:]}")
        return _strip_llama_cli_banner(text)


def load_llm(path: Path, n_ctx: int, n_threads: int, n_gpu_layers: int):
    try:
        import llama_cpp  # noqa: F401

        return LlamaCppBackend(path, n_ctx, n_threads, n_gpu_layers)
    except ImportError:
        cli = find_llama_cli()
        if cli is None:
            raise SystemExit(
                "Need llama-cpp-python or llama-cli.\n"
                "  python3 -m venv .venv && source .venv/bin/activate\n"
                "  pip install -r requirements.txt\n"
                "or put llama-cli on PATH (adtc/tools/llama.cpp/…/llama-cli)."
            )
        return LlamaCliBackend(path, cli, n_ctx, n_threads, n_gpu_layers)


def generate(llm, messages: list[dict], max_tokens: int, temperature: float) -> str:
    return llm.generate(messages, max_tokens, temperature)


def print_block(title: str, body: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}\n{body}\n{bar}", flush=True)


def run_prompts(llm, prompts: list[tuple[str, str]], max_tokens: int, temperature: float) -> None:
    for pid, text in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_turn(text)},
        ]
        t0 = time.perf_counter()
        reply = generate(llm, messages, max_tokens, temperature)
        elapsed = time.perf_counter() - t0
        print_block(f"{pid}  ({elapsed:.1f}s)\nPROMPT\n{text}", f"RESPONSE\n{reply}")


def interactive(llm, max_tokens: int, temperature: float) -> None:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("TebebAI tutor.  /reset  /quit", flush=True)
    while True:
        try:
            line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in {"/quit", "/exit", "q"}:
            break
        if low in {"/reset", "/clear"}:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(history cleared)", flush=True)
            continue
        messages.append({"role": "user", "content": user_turn(line)})
        t0 = time.perf_counter()
        reply = generate(llm, messages, max_tokens, temperature)
        messages.append({"role": "assistant", "content": reply})
        print(f"Tutor ({time.perf_counter() - t0:.1f}s)> {reply}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-metadata", action="store_true", help="Run metadata.json test_prompts and exit")
    ap.add_argument("--prompt", action="append", default=[], help="One-shot user prompt (repeatable)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_meta()
    path = resolve_gguf(meta)
    llm = load_llm(path, args.n_ctx, args.n_threads, args.n_gpu_layers)

    if args.from_metadata:
        rows = meta.get("test_prompts") or []
        if len(rows) != 2:
            raise SystemExit(f"expected exactly 2 test_prompts, got {len(rows)}")
        prompts = [(r.get("prompt_id", f"tp_{i:03d}"), r["prompt"]) for i, r in enumerate(rows, start=1)]
        run_prompts(llm, prompts, args.max_tokens, args.temperature)
        return

    if args.prompt:
        run_prompts(
            llm,
            [(f"prompt_{i}", p) for i, p in enumerate(args.prompt, start=1)],
            args.max_tokens,
            args.temperature,
        )
        return

    if not sys.stdin.isatty():
        raise SystemExit("Non-interactive stdin: pass --from-metadata or --prompt TEXT")

    interactive(llm, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
