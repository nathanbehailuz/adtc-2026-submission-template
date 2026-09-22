# Technical Report — TebebAI Offline English STEM Tutor

**Team ID:** TebebAI  
**Domain:** math_scientific_reasoning  
**Model:** tebeb_tutor_1.7b-Q4_K_M

---

## Problem

Students in low-connectivity settings need a **STEM tutor that runs fully offline** on an 8 GB budget laptop. The target user is a secondary-school learner (or peer tutor) working through English math and science without reliable internet, cloud APIs, or a discrete GPU.

TebebAI ships a single specialized GGUF that emphasizes tutoring behaviors—diagnose the first mistake, give one actionable hint, explain steps—rather than dumping final answers or leaking web-dataset markup (`####`, `<<>>`).

---

## Design Decisions

- **Base model:** `Qwen/Qwen3-1.7B` (Hub revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`). Fits laptop RAM after GGUF quantization; strong enough for multi-step arithmetic and middle-school science after SFT.
- **Adaptation:** QLoRA SFT (4-bit nf4 + LoRA **r=32 / α=64**, **2 epochs**) on English-only `sft_mix_v7` (**10636** rows): cleaned GSM8K + SciQ tutoring templates + **163** authored multi-constraint tutoring examples.
- **Quantization:** Converted f16→Q8/Q6/Q5/Q4 with llama.cpp **b10451**. **Deploy pick: Q4_K_M** — EN frozen accuracy matches Q5 within ~1 pp while targeting higher TPS / lower RSS on the Standard Laptop. Q5 remains a documented alternate.
- **Runtime:** llama.cpp only; one GGUF; no router; no network at inference.
- **Round-1 → Gate 2 fixes:** strip GSM8K `####` / `<<>>` from training targets; richer hint / first-error / multi-part scaffolding; authored tutoring bank; capacity bump (r 16→32, α 32→64, 1→2 epochs).

---

## Model Provenance

Must match `metadata.json` → `provenance`:

| Field | Value |
|-------|--------|
| Base model source | `huggingface:Qwen/Qwen3-1.7B` |
| Base model commit SHA | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Fine-tuning method | `qlora` |
| Training datasets | openai/gsm8k (cleaned); allenai/sciq (subset); authored_tutoring_v7 |

Proof-of-training: `provenance/` (adapter weights + config, train YAML, train/merge/convert scripts, dataset description + licenses + sample + SHA256 checksums, merge→GGUF notes, training log with mean `train_loss = 0.6026`).

Training data licenses: GSM8K (MIT); SciQ (CC-BY-NC 3.0); authored_tutoring_v7 (original); base `Qwen/Qwen3-1.7B` per Hub card.

### Before / after — what fine-tuning changed

**1) Held-out accuracy (reproducible)**

| Suite | v6 merged HF (prior SFT) | v7 merged HF | v7 Q4 GGUF |
|-------|-------------------------:|-------------:|-----------:|
| AfriMGSM EN | 0.392 | **0.444** | **0.428** |
| EN STEM holdout | 0.370 | **0.470** | **0.430** |
| Custom tutoring (soft) | 0.980 | **1.000** | **1.000** |

Q5 GGUF (EN-only eval): AfriMGSM EN **0.432**, EN STEM **0.430**, tutoring **1.0** — tied with Q4 for deploy choice.

**2) Prompt-level tutoring (Round-1 failure → Gate 2)**

Round-1 auto/judge prompts leaked GSM8K markup and dumped answers. Example **before** (Round-1 submitted model, stoich prompt):

> … 1.5 mol \* 44 g/mol = \<\<1.5\*44=66\>\>66 g of carbon dioxide is produced.  
> **#### 66**

**After** v7 Q4 judge smoke (`markup_leak_rate = 0.0` on 8 prompts): no `####` / `<<>>` in any reply. Example hint behavior (preview):

> Hint: subtract 7 from both sides. Why: undo addition, which keeps equality. Check question: what equation remains after that step? Do not state x.

Judge-smoke checklist pass rate **0.625** (5/8 full multi-part heuristics) — remaining gaps are scaffolding depth (analogy / exact stoich mass), not format leak.

**Laptop tutoring traces (v7 Q4, same texts as `metadata.json` `test_prompts`)**

We did **not** re-run unmodified stock Qwen3-1.7B on this laptop (no extra base GGUF in the submission). The quantitative base-vs-FT comparison is the accuracy table above (v6 SFT vs v7 SFT vs v7 Q4). Prompt-level before/after for tutoring *protocol* is Round-1 (markup leak / answer dump) vs v7 Q4 (no `####` / `<<>>`; hint and first-error without a final numeric dump on these two prompts):

*Prompt A (first-error):* “A student says a triangle with base 10 and height 4 has area 40. Identify the first mistake, then give one hint. Do not give the final number.”

> **v7 Q4:** They computed 10×4 without dividing by 2 (triangle area formula). Hint: recompute 10×4, then rewrite. Follow-up: what should the first intermediate be? No final area.

*Prompt B (hint):* “Betty needs $100 for a wallet. She already has half. Parents give $15; grandparents give twice that. Give one hint for how much more she needs.”

> **v7 Q4:** Hint: start by computing 15×2. Why: isolates the grandparents’ gift (near 30). Check question: what operation comes next? Do not state the final answer.

**3) Capacity / data change vs v6**

| Knob | v6 | v7 |
|------|----|----|
| LoRA r / α | 16 / 32 | **32 / 64** |
| Epochs | 1 | **2** |
| Markup in targets | present | **stripped** |
| Authored tutoring | thin | **163** multi-constraint rows |

---

## Constraints

- Target: **8 GB RAM**, integrated graphics, Ubuntu 22.04-class laptop
- Official scoring: **CPU llama.cpp only** (no CUDA requirement)
- Fully **offline** after `download_model.sh`
- Single artifact path in `metadata.json` `_runtime.model_path`

---

## Benchmarks

Self-reported development numbers. Official scores come from the ADTC profiler on the standard evaluation machine.

| Metric | Value |
|---|---|
| Machine (train / frozen eval) | Shadeform A6000-class (2026-09-22) |
| Machine (deploy target) | ADTC Standard Laptop (8 GB / 4 vCPU) |
| llama-bench tg (dev, Q4_K_M, 12 threads) | ~33.9 tok/s (b10451; not Standard Laptop) |
| Peak RSS / gen TPS (official) | Measured by the ADTC profiler on the Standard Laptop — not claimed from Shadeform llama-bench |
| Judge smoke (Q4) | markup_leak_rate **0.0**; checklist_pass_rate **0.625** (n=8) |
| Thermal throttling | Not observed on train host; laptop TBD |

v6 Jubail profiler reference (same base size, Q5_K_M): ~2.46 gen tok/s, ~1402 MB peak RSS — **not** claimed as v7 Gate 2 numbers.
