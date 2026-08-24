# TebebAI — ADTC 2026 Submission

Offline English STEM tutor for the **Africa Deep Tech Challenge 2026** Laptop LLM track (`math_scientific_reasoning`).

Based on the official [ADTC 2026 submission template](https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template).

**Model:** `tebeb_tutor_1.7b` · Qwen3-1.7B · llama.cpp · Q5_K_M (~1.2 GB)

Submit your repository URL via [adtc-2026.devpost.com](https://adtc-2026.devpost.com).

---

## Submission Checklist

Before submitting, confirm every item:

- [ ] Your repository is **public** on GitHub
- [ ] `metadata.json` is fully filled in — no placeholder values remain
- [ ] `metadata.json` contains exactly **2 test prompts** in the `test_prompts` array, written for your chosen domain
- [ ] `download_model.sh` successfully downloads your model to `model/`
- [ ] The downloaded file is a valid **GGUF format** (`.gguf`) weight file
- [ ] `model/*.gguf` is listed in `.gitignore` — do **not** commit large weight files
- [ ] `REPORT.md` is filled in with your technical writeup
- [ ] Running `bash download_model.sh` completes without errors
- [ ] Your model runs entirely **offline** — zero external network calls during inference

---

## Required File Structure

```
your-submission/
├── metadata.json          ← Required. Team, model, and test prompt metadata.
├── download_model.sh      ← Required. Downloads your .gguf model weight file.
├── REPORT.md              ← Required. Technical writeup (problem, design, benchmarks).
├── model/
│   └── tebeb_tutor_1.7b.gguf  ← Downloaded by the script above. Do NOT commit.
└── .gitignore             ← Must exclude *.gguf from version control.
```

Local demo helpers (not required by the evaluator):

```
├── chat.py                ← Terminal multi-turn chat with the tutor
└── requirements.txt       ← llama-cpp-python for a local venv
```

---

## Try the tutor (local demo)

```bash
cd adtc-2026-submission-template   # or this repo root if this *is* the submission
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash download_model.sh
python chat.py
```

`chat.py` checks that the GGUF exists (path from `metadata.json` → `_runtime.model_path`), loads it offline via llama.cpp, and keeps conversation history for follow-ups. Commands: `quit` / `exit` / `q`, `clear` / `reset`.

---

## metadata.json

Fill in every field. No field should remain at its placeholder value.

This submission uses domain `math_scientific_reasoning`, English-only (`language_scope: ["en"]`), and exactly two tutoring `test_prompts` (first-error + hint). See the file for current values.

### Field Reference

| Field | Required | Description |
|---|---|---|
| `team_id` | yes | Your unique team ID as registered on the ADTF portal |
| `domain` | yes | Challenge track (e.g. `math_scientific_reasoning`) |
| `language_scope` | yes | Array of BCP-47 language codes |
| `african_alpha_claim` | yes | `true` only if claiming the African Use Case Bonus |
| `budget_laptop_claim` | yes | Must be `true` — 8 GB RAM laptop profile |
| `submitter.*` | yes | Name, email, GitHub handle |
| `cross_disciplinary_pairing.*` | yes | Discipline + load-bearing description |
| `test_prompts` | yes | **Exactly 2** domain prompts |
| `model.runtime` | yes | Must be `llama.cpp` |
| `model.quantization` | yes | GGUF format (e.g. `Q5_K_M`) |
| `_runtime.model_path` | yes | Relative path to the `.gguf` (e.g. `model/tebeb_tutor_1.7b.gguf`) |

---

## download_model.sh

This script **must** download your model weight file to the `model/` directory.

Rules:

- Must be idempotent — safe to run multiple times without re-downloading.
- Must work without any credentials — your weights must be publicly accessible.
- The downloaded file path must exactly match `_runtime.model_path` in `metadata.json`.

---

## REPORT.md

Technical writeup for judges. Cover problem, design decisions, constraints, and benchmarks. See the filled [`REPORT.md`](REPORT.md) in this repo.

---

## Local Testing (profiler)

The ADTC profiler is open source:

```bash
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
```

```bash
bash download_model.sh

adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json \
  --skip-accuracy

cat submission.json
```

A valid run produces a `submission.json` with `"measured_on": "participant_laptop"`.

Profiler: [github.com/Africa-Deep-Tech-Foundation/adtc-profiler](https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler)

---

## Rules

1. **Public repository required.**
2. **No model weights in git.** Keep `*.gguf` / `model/` out of version control.
3. **100% offline during evaluation.**
4. **llama.cpp only** (GGUF).
5. **8 GB RAM limit** on the standard laptop profile.
6. **Exactly two** `test_prompts` in `metadata.json`.

---

## Support

Open an issue or contact the ADTF team at challenge@africadeeptech.org.

Eligibility: [adtc-2026.devpost.com/rules](https://adtc-2026.devpost.com/rules).

---

## License

This template is licensed under the terms of the [GNU GPL v3 License](LICENSE).
