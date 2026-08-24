# TebebAI: Offline STEM Tutor

**Team:** TebebAI
**ADTC 2026 Domain:** `math_scientific_reasoning`
**Model:** `tebeb_tutor_1.7b`
**Base:** `Qwen/Qwen3-1.7B`
**Runtime:** `llama.cpp`
**Quantization:** `Q5_K_M`
**Model size:** ~1.2 GB GGUF
**Cross-disciplinary pairing:** Education

---

## 1. Problem

TebebAI is an offline STEM tutor designed for students who cannot assume reliable internet access, cloud APIs, or high-end hardware.

The target is an ordinary **8 GB laptop with integrated graphics**. Within that constraint, the model should do more than return final answers. It should behave like a tutor by:

* solving STEM problems step by step,
* explaining concepts,
* giving hints without revealing the answer,
* and identifying the first mistake in a student's reasoning.

This creates a systems problem as much as a modeling problem. ADTC evaluates model quality, generation throughput, and memory efficiency, so we designed TebebAI around all three rather than optimizing accuracy alone.

The final system uses a single local GGUF model through `llama.cpp`. Once downloaded, it requires no API key, cloud service, or internet connection.

Two representative tutoring prompts are:

> "I wrote 3/4 + 1/2 = 4/6. Where did I go wrong? Give me one hint without telling me the final answer."

> "I'm stuck on 2x + 7 = 19. Can you give me one hint without telling me what x is?"

---

## 2. System Design

### Base model

We use **Qwen3-1.7B** as the base model.

A 1.7B parameter model gives us enough capacity for useful STEM reasoning while remaining small enough to quantize and run comfortably within the laptop memory budget.

| Decision              | Choice            | Reason                                                             |
| --------------------- | ----------------- | ------------------------------------------------------------------ |
| Base model            | `Qwen/Qwen3-1.7B` | Small enough for local deployment with a useful English STEM prior |
| Adaptation            | QLoRA SFT         | Efficient fine-tuning on limited training compute                  |
| Continued pretraining | Not used in v6    | Final English pipeline did not require CPT                         |
| Context length        | 2048              | Sufficient for the targeted tutoring interactions                  |
| Thinking mode         | Disabled          | Shorter responses and more predictable CPU inference               |
| Runtime               | `llama.cpp`       | Required GGUF-compatible local CPU inference                       |
| Final quant           | `Q5_K_M`          | Best measured balance of memory and throughput                     |

### Training data

The final supervised fine-tuning dataset contains **10,473 examples**.

| Source | Examples | Behaviors                         |
| ------ | -------: | --------------------------------- |
| GSM8K  |    7,473 | solve, explain, hint, first-error |
| SciQ   |    3,000 | solve, explain                    |

Instead of training only on question-answer pairs, we converted examples into different tutoring interactions. This lets the same model respond differently when a student asks for a full solution, an explanation, a hint, or help identifying a mistake.

Training data was deduplicated against the frozen English evaluation sets before training. No overlapping examples were found.

### Fine-tuning

The model was adapted using **QLoRA supervised fine-tuning** on an A100 GPU.

Key settings:

```text
LoRA rank:       16
LoRA alpha:      32
QLoRA:           4-bit NF4
Compute dtype:   bf16
Epochs:          1
Max sequence:    2048
```

After training, the LoRA adapter was merged into the original Qwen3-1.7B weights before conversion to GGUF.

---

## 3. Training and Deployment Pipeline

The complete pipeline is reproducible through:

```bash
cd adtc/hpc
bash submit_chain.sh
```

Pipeline:

```text
GSM8K + SciQ
      |
      v
10,473-example tutoring dataset
      |
      v
QLoRA supervised fine-tuning
      |
      v
Merge LoRA into Qwen3-1.7B
      |
      v
Convert merged model to GGUF
      |
      v
Q4 / Q5 / Q6 quantization sweep
      |
      v
Frozen evaluation + profiling
      |
      v
Q5_K_M deployment model
```

The Slurm workflow covers:

```text
prepare_mix
    ↓
train_sft
    ↓
merge_lora
    ↓
convert_gguf
    ↓
eval_hf / eval_gguf
    ↓
profile_gguf
```

This produces the final submission model:

```text
model/tebeb_tutor_1.7b.gguf
```

---

## 4. Quantization

The fine-tuned model was converted to GGUF and tested at multiple quantization levels.

| Quantization |  Gen TPS |       Peak RSS | Hardware Composite |
| ------------ | -------: | -------------: | -----------------: |
| Q4_K_M       |     2.54 |     1897.66 MB |              19.79 |
| **Q5_K_M**   | **2.46** | **1402.11 MB** |          **21.01** |
| Q6_K         |     2.44 |     1553.22 MB |              20.55 |

We selected **Q5_K_M**.

Q4 was slightly faster in our run, but Q5 used substantially less peak memory and produced the highest measured hardware composite score. Q6 increased memory use without a meaningful throughput advantage.

The final model is approximately **1.2 GB on disk**.

QLoRA's 4-bit training representation and the final Q5 GGUF are separate optimization stages. The former reduces the cost of fine-tuning, while the latter determines the deployment trade-off between model quality, memory, and CPU performance.

---

## 5. Evaluation

We evaluate different parts of the system separately because a single accuracy number would hide important differences between STEM problem solving and tutoring behavior.

### Model quality

| Evaluation           |   n |     Score | Measures                                    |
| -------------------- | --: | --------: | ------------------------------------------- |
| Custom tutoring set  | 101 | **98.0%** | Tutoring behavior and instruction following |
| AfriMGSM English     | 250 | **39.2%** | Unseen mathematical problem solving         |
| English STEM holdout | 100 | **37.0%** | Unseen STEM problem solving                 |
| AfriMGSM Amharic     | 250 |      2.4% | Secondary multilingual diagnostic           |
| AfriMMLU Amharic     | 500 |     21.6% | Secondary multilingual diagnostic           |

The **98% tutoring score should not be interpreted as 98% general STEM accuracy**. The evaluations measure different capabilities.

Across the current evaluation suite, results range widely depending on the task. TebebAI performs considerably better at following the tutoring behaviors it was explicitly trained for than at solving every unseen reasoning problem correctly.

Improving raw STEM accuracy while preserving those tutoring behaviors is a major next step.

### Deployment performance

For the selected Q5_K_M model:

| Metric                      |        Result |
| --------------------------- | ------------: |
| Generation throughput       | 2.46 tokens/s |
| Peak RSS                    |       1402 MB |
| Steady RSS                  |       1304 MB |
| Thermal throttling observed |            No |
| Performance score           |          16.4 |
| Memory-efficiency score     |         80.44 |
| Hardware composite          |         21.01 |

These measurements were collected on **NYU Abu Dhabi Jubail compute nodes**, not yet on a physical ADTC Standard Laptop.

The memory result provides substantial headroom below the competition's 7 GB evaluator budget, but throughput and thermal behavior still need to be validated on the exact target hardware.

---

## 6. Tools

| Tool                       | Purpose                                                |
| -------------------------- | ------------------------------------------------------ |
| **PyTorch / Transformers** | Base-model loading and fine-tuning                     |
| **PEFT / QLoRA**           | Parameter-efficient supervised fine-tuning             |
| **Hugging Face Datasets**  | Loading and preprocessing GSM8K and SciQ               |
| **llama.cpp**              | GGUF conversion and local CPU inference                |
| **ADTC Profiler**          | Throughput, memory, and hardware evaluation            |
| **Slurm**                  | Reproducible training and evaluation jobs on NYUAD HPC |

The pipeline was intentionally kept simple. TebebAI uses one model and one inference runtime rather than a router, retrieval service, cloud fallback, or secondary model.

---

## 7. Constraints and Limitations

### Target hardware

TebebAI is designed around the ADTC Standard Laptop:

* 8 GB DDR4 RAM
* x86-64 CPU
* integrated graphics
* Ubuntu 22.04
* no discrete GPU required for inference

The final Q5 model currently peaks at approximately **1.4 GB RSS** in our profiler run.

### Offline operation

Inference requires no network access.

The model is downloaded once using:

```bash
bash download_model.sh
```

After that, TebebAI runs entirely from the local GGUF file.

### Current model quality

The current evaluation results vary substantially by task, from roughly the mid-30% range on unseen English STEM reasoning benchmarks to much stronger scores on the tutoring-behavior rubric.

This means the system is currently better at **how it tutors** than it is at solving every difficult STEM problem correctly.

### Multilingual support

TebebAI originally explored bilingual Amharic-English tutoring.

Earlier experiments encountered poor Amharic tokenizer efficiency, limited high-quality tutoring data, and weak evaluation performance. We therefore narrowed v6 to English rather than shipping an unreliable multilingual model.

The Amharic work remains part of the next stage of the project.

---

## 8. Running TebebAI

Download the model:

```bash
bash download_model.sh
```

Then launch the local terminal interface:

```bash
python chat.py
```

The model loads from:

```text
model/tebeb_tutor_1.7b.gguf
```

`chat.py` provides a minimal multi-turn interface and removes Qwen thinking markers before displaying responses.

No internet connection or API credentials are required during inference.

---

## 9. What We Learned

### Specialization helps small models

A 1.7B model cannot match large cloud models across every reasoning task, but targeted fine-tuning can shape it into a more useful specialist.

The tutoring dataset noticeably improved behaviors such as giving hints, explaining steps, and identifying mistakes.

### Deployment has to be optimized separately

Fine-tuning efficiency does not automatically produce efficient inference.

QLoRA reduced training memory requirements, but choosing the final deployment format still required a separate Q4/Q5/Q6 quantization sweep.

### Scope matters

Earlier versions explored a larger bilingual system. Those experiments were useful, but narrowing v6 to a reproducible English model allowed us to complete and evaluate the full deployment pipeline.

---

## 10. Next Steps

### Improve STEM accuracy

The current evaluation suite produces scores ranging from roughly **34% to 98% depending on what is being measured**.

We plan to expand the frozen evaluation suite and clearly separate:

* answer accuracy,
* reasoning quality,
* tutoring behavior,
* and instruction following.

We also want to benchmark TebebAI directly against the original Qwen3-1.7B model to quantify the effect of fine-tuning.

### Validate on target hardware

The next profiling run will use a physical **8 GB ADTC Standard Laptop** and measure:

* generation throughput,
* peak memory,
* CPU utilization,
* temperature,
* thermal throttling,
* and accuracy within the official profiler.

### Return to Amharic

A future version will revisit Amharic using better native tutoring data and a base model selected with tokenizer efficiency in mind.

### Build a simple offline interface

The current terminal interface proves the deployment path. The next product layer will provide a lightweight local chat interface that hides model configuration from the student.

### Test with students

The longer-term goal is to evaluate whether hinting, explanation, and error diagnosis actually improve learning outcomes in environments where cloud access cannot be assumed.

---

## Reproducibility

Training, evaluation, and profiling artifacts are stored under:

```text
adtc/docs/artifacts/v6/
```

Detailed results:

```text
adtc/docs/RESULTS_REPORT.md
```

Training configuration:

```text
adtc/training/configs/qlora_qwen3_1_7b_v6.yaml
```

End-to-end pipeline:

```text
adtc/hpc/submit_chain.sh
```

---

*This research was carried out using the High Performance Computing resources at New York University Abu Dhabi.*
