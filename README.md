# From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs

## Environment

```bash
conda create -n conflict python==3.10 -y
conda activate conflict
pip install -r requirements.txt
```

## Project Structure

```
.
├── methods/                  # Unified decoding method implementations
│   ├── inference.py          # Single-GPU inference entry point
│   ├── base.py               # Abstract base class for decoding methods
│   ├── greedy.py / cad.py / adacad.py / coiecd.py / cocoa.py
│   ├── simple_interp.py      # Fixed-α interpolation
│   └── arr.py                # ARR (Adaptive Regime Routing)
├── evaluation/               # Unified evaluation framework
│   ├── unified_eval.py       # Evaluation entry point
│   ├── metrics/
│   │   └── qa_metrics.py     # EM / Substring EM / F1
│   └── utils.py              # Answer normalization, data loading
├── CAD/ / AdaCAD/ / CoCoA/ / COIECD/   # Original baseline code (dual-GPU)
├── data/                     # Unified-format benchmark data
├── inference_qa_self.sh      # Batch QA inference (single-GPU methods)
├── inference_qa_origin.sh    # Batch QA inference (original dual-GPU baselines)
├── inference_tristate_self.sh    # Tristate benchmark (single-GPU)
├── inference_tristate_origin.sh  # Tristate benchmark (dual-GPU)
└── requirements.txt
```

---

## Inference

Inference is split into two categories: **single-GPU methods** (via `methods/inference.py`) and **original baseline dual-GPU methods** (via each sub-repo's `run_group_decode_fileio.sh`).

### Data Format

All benchmarks use a unified **dual-process JSONL** format with two lines per sample:

```jsonl
{"input_index": 0, "assigned_process": 0, "context_string": "<context+question>", "assigned_weight": 2, "gold_answers": "answer", "benchmark": "nq_swap", "task_type": "qa"}
{"input_index": 0, "assigned_process": 1, "context_string": "<question_only>", "assigned_weight": -1, "gold_answers": "answer", "benchmark": "nq_swap", "task_type": "qa"}
```

- `assigned_process=0`: input with context (context prompt)
- `assigned_process=1`: input without context (prior prompt)

### Single-GPU Inference

Use `methods/inference.py`. Supported methods: `greedy`, `greedy_no_ctx`, `cad`, `coiecd`, `simple_interp`, `adacad`, `cocoa`, `arr`.

```bash
# Greedy baseline
python -m methods.inference --method greedy \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/nq_swap.jsonl \
    --output_path results/greedy/Meta-Llama-3-8B/nq_swap.jsonl \
    --max_new_tokens 32

# COIECD (entropy-constrained decoding)
python -m methods.inference --method coiecd \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/nq_swap.jsonl \
    --output_path results/coiecd/Meta-Llama-3-8B/nq_swap.jsonl \
    --max_new_tokens 32 --alpha 1.0

# Simple Interpolation (fixed α)
python -m methods.inference --method simple_interp \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/nq_swap.jsonl \
    --output_path results/simple_interp/Meta-Llama-3-8B/nq_swap.jsonl \
    --max_new_tokens 32 --interp_alpha 0.75

# ARR (Adaptive Regime Routing)
python -m methods.inference --method arr \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/nq_swap.jsonl \
    --output_path results/arr/Meta-Llama-3-8B/nq_swap.jsonl \
    --max_new_tokens 32
```

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Decoding method name | (required) |
| `--model` | HuggingFace model ID or local path | (required) |
| `--input_path` | Dual-process JSONL input file | (required) |
| `--output_path` | Prediction output path | (required) |
| `--max_new_tokens` | Maximum generation tokens | 32 |
| `--max_ctx_len` | Input truncation length | 4064 |
| `--alpha` | COIECD/CAD α parameter | 1.0 |
| `--interp_alpha` | SimpleInterp α (context weight) | 0.75 |
| `--dtype` | Model precision (float16/bfloat16/float32) | float16 |

Inference supports **resumption**: already-completed `input_index` entries are automatically skipped.

### Original Baseline Dual-GPU Inference (CAD / AdaCAD / CoCoA)

These methods use the accelerate dual-process implementation in their respective sub-repos and require 2 GPUs:

```bash
export MODEL_NAME=/data/models/Meta-Llama-3-8B

# CAD
cd CAD && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/nq_swap.jsonl" \
    4096 4064 32 0.0 \
    ../results/cad/Meta-Llama-3-8B/nq_swap.jsonl

# AdaCAD
cd AdaCAD && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/nq_swap.jsonl" \
    4096 4064 32 0.0 \
    2 no 0 ../results/adacad/Meta-Llama-3-8B/nq_swap.jsonl

# CoCoA
cd CoCoA && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/nq_swap.jsonl" \
    4096 4064 32 0.0 \
    2 no 0 ../results/cocoa/Meta-Llama-3-8B/nq_swap.jsonl
```

### Batch Inference Scripts

Four shell scripts are provided for batch inference:

```bash
# Single-GPU methods on QA benchmarks
bash inference_qa_self.sh
# Controlled via environment variables:
#   MODEL_NAME  — model path (default: /data/models/Meta-Llama-3-8B)
#   METHODS     — method list (default: "arr")
#   BENCHMARKS  — dataset list (default: "nq tabmwp triviaqa hotpotqa")
#   GPU         — GPU index (default: 0)
#   OUTPUT_ROOT — output root directory (default: results_new)
#   OVERWRITE   — set to 1 to force re-run

# Original dual-GPU methods on QA benchmarks
bash inference_qa_origin.sh
# Additional env var: DEVICE="0,1"

# Single-GPU methods on tristate benchmark
bash inference_tristate_self.sh
# Data from: facts_repo/{MODEL_SHORT}/data/{C_right_P_wrong,C_right_P_right,C_wrong_P_right}.jsonl

# Dual-GPU methods on tristate benchmark
bash inference_tristate_origin.sh
```

Scripts include built-in **input filtering** (drops samples exceeding `max_ctx_len` or missing pairs) and **checkpoint detection** (skips if complete output already exists).

### Parameter Reference

| Parameter | QA Task |
|-----------|---------|
| `max_new_tokens` | 32 |
| `top_p` | 0.0 (greedy) |
| `max_ctx_len` | 4064 |
| `seed` | 42 |

---

## Evaluation

Use the unified evaluation entry point `evaluation/unified_eval.py` for QA tasks.

### QA Evaluation

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/nq_swap.jsonl \
    --pred_path results/greedy/Meta-Llama-3-8B/nq_swap.jsonl \
    --metrics em,f1,substring_em
```

QA metrics:

| Metric | Description |
|--------|-------------|
| `em` | Exact Match (SQuAD-style normalization) |
| `em_test` | EM (preserving punctuation) |
| `substring_em` | Gold answer is a substring of the prediction |
| `f1` | Token-level F1 score |

### Output Format

Results are output as JSON and can be saved via `--output_path`:

```json
{
  "em": 0.4523,
  "f1": 0.5812,
  "substring_em": 0.6134,
  "total": 4000,
  "missing": 0
}
```
