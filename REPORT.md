# TebebAI — Offline English STEM Tutor

**Team:** TebebAI · ADTC 2026 · domain `math_scientific_reasoning`  
**Model:** `tebeb_tutor_1.7b` · base `Qwen/Qwen3-1.7B` · runtime **llama.cpp** · quant **Q5_K_M** (~1.2 GB GGUF)  
**Cross-disciplinary pairing:** education (load-bearing) — offline STEM tutoring for low-connectivity learners

---

## 1. Problem

Students in low-connectivity settings need a **STEM tutor that runs fully offline** on an ordinary 8 GB laptop—not a cloud chat they cannot reach. The product must behave like a teacher: **solve**, **explain**, **hint without revealing the answer**, and **diagnose the first error** in student work. ADTC’s Laptop LLM track scores accuracy (50%), generation throughput (30%), and memory efficiency (20%) under a ~7 GB peak-RSS budget, with **llama.cpp / GGUF only**.

**TebebAI** (*tebeb* = wisdom that is shared) started as a bilingual Amharic–English ambition. Weeks of v0–v5 experiments on Jubail hit hard walls: gated Hugging Face corpora, missing AfriqueLLM GSM8K access, severe Amharic tokenizer fertility on Qwen vs Gemma, and weak Amharic eval (AfriMGSM AM remained ~2%). We deleted ~365 GB of dead-end artifacts and shipped a focused **English-only v6** specialist so something real reaches students within the hackathon window. The bilingual vision remains the next iteration; scope discipline was the shippable choice.

Target behaviors (and our two `metadata.json` test prompts):

- *“I wrote 3/4 + 1/2 = 4/6. Where did I go wrong? Give me one hint without telling me the final answer.”*
- *“I'm stuck on 2x + 7 = 19. Can you give me one hint without telling me what x is?”*

One GGUF file, no API keys, no multi-model router: download once, tutor offline (`bash download_model.sh` → `model/tebeb_tutor_1.7b.gguf` from Hugging Face `nz2212/tebeb_tutor_1.7b`; optional local demo: `python chat.py`).

---

## 2. Design Decisions

### Base model and adaptation

| Item | Choice | Why |
|------|--------|-----|
| Base | `Qwen/Qwen3-1.7B` | Fits laptop RSS after PTQ; strong English STEM prior; official GGUF / llama.cpp path |
| Adaptation | **QLoRA SFT only** (no CPT in v6) | Parameter-efficient on A100; CPT deferred after multilingual cut |
| Train config | `lora_r=16`, `alpha=32`, 4-bit NF4, bf16, 1 epoch, max seq 2048 | See `adtc/training/configs/qlora_qwen3_1_7b_v6.yaml` |
| Thinking mode | Off for eval/deploy | HF: `enable_thinking=False`; GGUF: `/no_think` prefix — shorter, more consistent replies |

### Data (`adtc/data/`)

Built `sft_mix_v6.jsonl` (**10,473** rows) via `mix_sft_v6.py`:

| Source | n | Behaviors |
|--------|--:|-----------|
| GSM8K train | 7,473 | solve, explain, hint, first_error |
| SciQ train | 3,000 | solve, explain |

Training rows were **deduplicated** against frozen eval sets `en_stem_holdout_v0` and `afrimgsm_eng_test_v0` (**0** leakage drops). Frozen eval (never trained on): AfriMGSM EN (250), EN STEM holdout (100), custom tutoring rubric (101); Amharic suites kept only as secondary diagnostics.

### Pipeline (`adtc/hpc/submit_chain.sh`)

Reproduced end-to-end on NYUAD Jubail HPC:

```
GSM8K + SciQ  →  sft_mix_v6 (10,473)
       ↓
QLoRA SFT (A100, bf16)  →  merge LoRA into base
       ↓
GGUF f16 → Q8 / Q6 / Q5 / Q4
       ↓
HF frozen eval  +  GGUF frozen eval  +  ADTC profiler
       ↓
Deploy pick: Q5_K_M  →  this submission package
```

Slurm stages: `prepare_mix` → `train_sft` → `merge_lora` → `convert_gguf` → `eval_hf` / `eval_gguf` → `profile_gguf`. Staging helpers live under `adtc/eval/`; the submission surface is this folder (`metadata.json`, `download_model.sh`, `REPORT.md`, `chat.py`).

### Quantization pick (Gate 5)

We swept Q4_K_M / Q5_K_M / Q6_K with the pinned ADTC profiler (participant mode, hardware telemetry). **Q5_K_M** won the Pareto trade-off: ~1.2 GB on disk, acceptable TPS, **lowest peak RSS** among scored quants, highest hardware composite under our skip-accuracy gate.

| Quant | Gen TPS | Peak RSS (MB) | Composite |
|-------|--------:|--------------:|----------:|
| Q4_K_M | 2.54 | 1897.66 | 19.79 |
| **Q5_K_M** | **2.46** | **1402.11** | **21.01** |
| Q6_K | 2.44 | 1553.22 | 20.55 |

Shipped name: `tebeb_tutor_1.7b` at `_runtime.model_path` = `model/tebeb_tutor_1.7b.gguf`.

---

## 3. Constraints

- **Runtime:** llama.cpp + single GGUF only (ADTC rule).
- **Hardware:** 8 GB Standard Laptop profile; peak RSS must stay under the ~7 GB evaluator budget. Our deploy peak is **~1.4 GB**.
- **Offline:** zero network during inference; weights arrive only via `download_model.sh` before profiling.
- **Score shape:** `0.5·S_acc + 0.3·S_tps + 0.2·S_mem` (with thermal penalty / OOM disqualification in the official profiler).
- **Training vs deploy precision:** QLoRA trains in 4-bit; deployment is **Q5_K_M** GGUF—separate optimization problems.
- **Honest caveat:** profiler TPS / RSS / `S_tps=16.4` / `S_mem=80.44` were measured on **Jubail compute nodes** (EPYC), not yet re-run on a physical 8 GB Standard Laptop. Gate 5 used `--skip-accuracy`; accuracy below is from frozen HF/GGUF eval, not the profiler accuracy arrays.

---

## 4. Benchmarks

### HF frozen eval (merged checkpoint, full sets)

| Suite | n | Accuracy |
|-------|--:|---------:|
| Custom tutoring (hint / first-error / explain) | 101 | **0.980** |
| AfriMGSM EN | 250 | **0.392** |
| EN STEM holdout | 100 | **0.370** |
| AfriMGSM AM (secondary; not trained) | 250 | 0.024 |
| AfriMMLU AM (secondary) | 500 | 0.216 |

Primary EN KPIs improved vs our early bilingual baseline (~0.34 on AfriMGSM EN and holdout) while keeping the tutoring rubric near ceiling—the product behavior we optimized for.

### Profiler (deploy GGUF Q5_K_M)

| Metric | Value |
|--------|------:|
| Generation tokens/s | 2.46 |
| Peak RSS | 1402 MB |
| Steady RSS | 1304 MB |
| Throttled | false |
| S_tps (performance) | 16.4 |
| S_mem (efficiency) | 80.44 |
| Hardware composite | 21.01 |

Artifacts and day-to-day notes live in the research tree under `adtc/docs/artifacts/v6/` and `adtc/docs/RESULTS_REPORT.md`. Local smoke after download: `python chat.py` (strips `<think>` / `<<…>>`, keeps multi-turn context) or the pinned `adtc-profiler` participant run documented in `README.md`.

---

## 5. What we learned / next

**Small specialist + tutoring data beats oversized multilingual ambition under laptop constraints.** A 1.7B model with 10k curated rows nails hint and first-error behavior even when raw MGSM sits near ~40%.

**Next:** native Amharic tutoring SFT (authored data, fertility-aware base), re-profile on a real 8 GB laptop with accuracy-inclusive profiler, thin offline UI, and a field pilot measuring learning—not just homework completion.

*This research was carried out on the High Performance Computing resources at New York University Abu Dhabi.*
