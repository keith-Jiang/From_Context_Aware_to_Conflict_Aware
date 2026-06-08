#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MODEL_NAME=/data/models/Meta-Llama-3-8B
export MODEL_NAME
MODEL_SHORT=$(basename "$MODEL_NAME")
if [ -z "${PYTHON:-}" ]; then
    if [ -x /home/linux/anaconda3/envs/OptCAD/bin/python ]; then
        PYTHON=/home/linux/anaconda3/envs/OptCAD/bin/python
    else
        PYTHON=python
    fi
fi
if [ -x /home/linux/anaconda3/envs/OptCAD/bin/python ]; then
    export PATH="/home/linux/anaconda3/envs/OptCAD/bin:$PATH"
fi
export PYTHON
GPU=${GPU:-0}
MAX_TOKENS=${MAX_TOKENS:-32}
MAX_CTX_LEN=${MAX_CTX_LEN:-4064}
OUTPUT_ROOT=${OUTPUT_ROOT:-results_new}
OVERWRITE=${OVERWRITE:-0}
# METHODS=${METHODS:-"greedy coiecd simple_interp arr"}
METHODS=${METHODS:-"arr"}
BENCHMARKS=${BENCHMARKS:-"nq tabmwp triviaqa hotpotqa"}
COIECD_ALPHA=${COIECD_ALPHA:-1.0}
SIMPLE_INTERP_ALPHA=${SIMPLE_INTERP_ALPHA:-0.75}
COCOA_ALPHA_RENYI=${COCOA_ALPHA_RENYI:-0.5}
COCOA_Z=${COCOA_Z:-5.0}
COCOA_GAMMA=${COCOA_GAMMA:-1.0}
COCOA_DELTA=${COCOA_DELTA:-1e-8}

prepare_filtered_input() {
    local source_input=$1
    local filtered_input=$2
    local max_ctx_len=$3
    local tmp_filtered="${filtered_input}.tmp.$$"

    mkdir -p "$(dirname "$filtered_input")"
    "$PYTHON" - "$source_input" "$tmp_filtered" "$MODEL_NAME" "$max_ctx_len" <<'PY'
import json
import sys
from collections import OrderedDict

from transformers import AutoTokenizer

source_path, filtered_path, model_name, max_ctx_len = sys.argv[1:]
max_ctx_len = int(max_ctx_len)

tokenizer = AutoTokenizer.from_pretrained(model_name)
rows = []
groups = OrderedDict()
with open(source_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows.append(row)
        idx = row["input_index"]
        groups.setdefault(idx, {})[row.get("assigned_process")] = row

allowed = set()
drop_missing_pair = 0
drop_too_long = 0
for idx, group in groups.items():
    if 0 not in group or 1 not in group:
        drop_missing_pair += 1
        continue
    keep = True
    for proc in (0, 1):
        text = group[proc].get("context_string", "")
        token_len = len(tokenizer.encode(text, add_special_tokens=True))
        if token_len > max_ctx_len:
            keep = False
            break
    if keep:
        allowed.add(idx)
    else:
        drop_too_long += 1

with open(filtered_path, "w", encoding="utf-8") as f:
    for row in rows:
        if row["input_index"] in allowed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(
    f"Filtered input: kept {len(allowed)}/{len(groups)} samples "
    f"(too_long={drop_too_long}, missing_pair={drop_missing_pair}) -> {filtered_path}"
)
PY
    mv -f "$tmp_filtered" "$filtered_input"
}

filter_existing_predictions() {
    local input=$1
    local output=$2

    [ -f "$output" ] || return 1

    "$PYTHON" - "$input" "$output" <<'PY'
import json
import sys

input_path, output_path = sys.argv[1:]

allowed_order = []
with open(input_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if item.get("assigned_process") == 0:
            allowed_order.append(item["input_index"])
allowed = set(allowed_order)

row_by_idx = {}
valid = True
with open(output_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            valid = False
            continue
        idx = item.get("input_index", item.get("id"))
        if idx in allowed and idx not in row_by_idx:
            row_by_idx[idx] = item

with open(output_path, "w", encoding="utf-8") as f:
    for idx in allowed_order:
        if idx in row_by_idx:
            f.write(json.dumps(row_by_idx[idx], ensure_ascii=False) + "\n")

missing = [idx for idx in allowed_order if idx not in row_by_idx]
if missing or not valid:
    print(f"    Existing output incomplete after filtering: kept {len(row_by_idx)}/{len(allowed_order)}")
    sys.exit(1)

print(f"    Existing output matches filtered input: {len(row_by_idx)} samples")
PY
}

echo "Model: $MODEL_NAME"
echo "Model short: $MODEL_SHORT"
echo "Python: $PYTHON"
echo "Self GPU: $GPU"
echo "Methods: $METHODS"
echo "Benchmarks: $BENCHMARKS"
echo "Max tokens: $MAX_TOKENS"
echo "Output root: $OUTPUT_ROOT"
echo "Overwrite: $OVERWRITE"
echo ""

run_self_method() {
    local method=$1
    local input=$2
    local output=$3

    if [ "$OVERWRITE" != "1" ] && filter_existing_predictions "$input" "$output"; then
        echo "  -> $method: $output exists, skipping"
        return
    fi

    mkdir -p "$(dirname "$output")"
    if [ "$OVERWRITE" = "1" ]; then
        : > "$output"
    fi

    local extra_args=""
    case $method in
        coiecd)
            extra_args="--alpha $COIECD_ALPHA"
            ;;
        simple_interp)
            extra_args="--interp_alpha $SIMPLE_INTERP_ALPHA"
            ;;
        cocoa)
            extra_args="--cocoa_alpha_renyi $COCOA_ALPHA_RENYI --cocoa_z $COCOA_Z --cocoa_gamma $COCOA_GAMMA --cocoa_delta $COCOA_DELTA"
            ;;
        arr)
            extra_args=""
            ;;
    esac

    echo "  -> Running $method with unified implementation..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m methods.inference \
        --method "$method" \
        --model "$MODEL_NAME" \
        --input_path "$input" \
        --output_path "$output" \
        --max_new_tokens "$MAX_TOKENS" \
        --max_ctx_len "$MAX_CTX_LEN" \
        $extra_args
}

for BENCH in $BENCHMARKS; do
    INPUT="$SCRIPT_DIR/data/${BENCH}.jsonl"

    if [ ! -f "$INPUT" ]; then
        echo "Skipping $BENCH: $INPUT not found"
        continue
    fi

    FILTERED_INPUT="$SCRIPT_DIR/${OUTPUT_ROOT}/filtered_inputs/${MODEL_SHORT}/qa_maxctx${MAX_CTX_LEN}/${BENCH}.jsonl"
    prepare_filtered_input "$INPUT" "$FILTERED_INPUT" "$MAX_CTX_LEN"

    if [ ! -s "$FILTERED_INPUT" ]; then
        echo "Skipping $BENCH: filtered input is empty"
        continue
    fi

    echo "========== ${BENCH} Inference =========="

    for method in $METHODS; do
        OUTPUT="$SCRIPT_DIR/${OUTPUT_ROOT}/${method}/${MODEL_SHORT}/${BENCH}.jsonl"

        case $method in
            greedy|coiecd|simple_interp|cocoa|arr)
                run_self_method "$method" "$FILTERED_INPUT" "$OUTPUT"
                ;;
            *)
                echo "  -> Unknown self method: $method, skipping"
                ;;
        esac
    done

    echo "========== ${BENCH} Done =========="
    echo ""
done

echo "All QA self inference tasks completed."
