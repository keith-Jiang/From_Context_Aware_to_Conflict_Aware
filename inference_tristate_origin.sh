#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MODEL_SHORT=${MODEL_SHORT:-Meta-Llama-3-8B}
MODEL_NAME=/data/models/$MODEL_SHORT
export MODEL_NAME MODEL_SHORT
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
SEED=${SEED:-42}
ORIGIN_GLOBALLEN=${ORIGIN_GLOBALLEN:-4096}
ORIGIN_MAXCTXLEN=${ORIGIN_MAXCTXLEN:-4064}
MAX_TOKENS=${MAX_TOKENS:-32}
ORIGIN_TOPP=${ORIGIN_TOPP:-0.0}
GPUS=${GPUS:-2}
INT4=${INT4:-no}
THRESHOLD=${THRESHOLD:-0}
DEVICE=${DEVICE:-"0,1"}
OUTPUT_ROOT=${OUTPUT_ROOT:-results_new}
OVERWRITE=${OVERWRITE:-0}
METHODS=${METHODS:-"cad_0.25 cad_0.5 cad_0.75 cad adacad cocoa"}
BENCHMARKS=${BENCHMARKS:-"C_right_P_wrong C_right_P_right C_wrong_P_right"}

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

echo "Model: $MODEL_NAME"
echo "Model short: $MODEL_SHORT"
echo "Python: $PYTHON"
echo "Origin devices: $DEVICE"
echo "Methods: $METHODS"
echo "Benchmarks: $BENCHMARKS"
echo "Output root: $OUTPUT_ROOT"
echo "Overwrite: $OVERWRITE"
echo ""

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

convert_origin_qa_output() {
    local raw_output=$1
    local final_output=$2
    local bench=$3
    local method=$4
    local input=$5

    "$PYTHON" - "$raw_output" "$final_output" "$bench" "$method" "$input" <<'PY'
import json
import sys

raw_path, final_path, bench, method_name, input_path = sys.argv[1:]

gold_map = {}
with open(input_path, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        idx = item["input_index"]
        if idx not in gold_map:
            gold_map[idx] = item

results = []
with open(raw_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if item.get("assigned_process", -1) != 0:
            continue
        idx = item["input_index"]
        strings = item.get("string", "")
        prediction = strings[0] if isinstance(strings, list) else str(strings)
        gold = gold_map.get(idx, {})
        results.append({
            "input_index": idx,
            "benchmark": gold.get("benchmark", bench),
            "task_type": gold.get("task_type", "qa"),
            "gold_answers": gold.get("gold_answers", ""),
            "prediction": prediction.strip(),
            "method": method_name,
        })

results.sort(key=lambda x: x["input_index"])
with open(final_path, "w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"    Converted {len(results)} predictions -> {final_path}")
PY
}

rewrite_assigned_weights() {
    local source_input=$1
    local rewritten_input=$2
    local cad_alpha=$3

    "$PYTHON" - "$source_input" "$rewritten_input" "$cad_alpha" <<'PY'
import json
import sys

source_path, rewritten_path, cad_alpha = sys.argv[1:]
cad_alpha = float(cad_alpha)
tau = 1.0 + cad_alpha

with open(source_path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

for row in rows:
    if row.get("assigned_process") == 0:
        row["assigned_weight"] = tau
    else:
        row["assigned_weight"] = -cad_alpha

with open(rewritten_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"    Rewrote assigned_weights: tau={tau}, alpha={cad_alpha} -> {rewritten_path}")
PY
}

run_origin_method() {
    local method=$1
    local input=$2
    local output=$3
    local cad_alpha=${4:-}
    local raw_output="${output%.jsonl}_raw.jsonl"
    local repo_dir=""
    local actual_input="$input"

    if [ "$OVERWRITE" != "1" ] && filter_existing_predictions "$input" "$output"; then
        echo "  -> $method: $output exists, skipping"
        return
    fi

    local base_method="$method"
    case $method in
        cad_*) base_method="cad" ;;
        adacad) ;;
        cocoa) ;;
        cad) ;;
        *)
            echo "  -> Unknown original method: $method, skipping"
            return
            ;;
    esac

    case $base_method in
        cad) repo_dir="CAD" ;;
        adacad) repo_dir="AdaCAD" ;;
        cocoa) repo_dir="CoCoA" ;;
    esac

    mkdir -p "$(dirname "$raw_output")"

    if [ -n "$cad_alpha" ]; then
        actual_input="${input%.jsonl}_cad_alpha${cad_alpha}.jsonl"
        rewrite_assigned_weights "$input" "$actual_input" "$cad_alpha"
    fi

    echo "  -> Running $method (base=$base_method, alpha=$cad_alpha) with original repo..."

    case $base_method in
        cad)
            (
                cd "$repo_dir"
                bash run_group_decode_fileio.sh \
                    "$SEED" "$DEVICE" "fin|$actual_input" \
                    "$ORIGIN_GLOBALLEN" "$ORIGIN_MAXCTXLEN" "$MAX_TOKENS" "$ORIGIN_TOPP" \
                    "$raw_output"
            )
            ;;
        adacad|cocoa)
            (
                cd "$repo_dir"
                bash run_group_decode_fileio.sh \
                    "$SEED" "$DEVICE" "fin|$actual_input" \
                    "$ORIGIN_GLOBALLEN" "$ORIGIN_MAXCTXLEN" "$MAX_TOKENS" "$ORIGIN_TOPP" \
                    "$GPUS" "$INT4" "$THRESHOLD" "$raw_output"
            )
            ;;
    esac

    convert_origin_qa_output "$raw_output" "$output" "$(basename "$input" .jsonl)" "$method" "$input"
}

for BENCH in $BENCHMARKS; do
    INPUT="$SCRIPT_DIR/facts_repo/${MODEL_SHORT}/data/${BENCH}.jsonl"

    if [ ! -f "$INPUT" ]; then
        echo "Skipping $BENCH: $INPUT not found"
        continue
    fi

    FILTERED_INPUT="$SCRIPT_DIR/${OUTPUT_ROOT}/filtered_inputs/${MODEL_SHORT}/qa_maxctx${ORIGIN_MAXCTXLEN}/${BENCH}.jsonl"
    prepare_filtered_input "$INPUT" "$FILTERED_INPUT" "$ORIGIN_MAXCTXLEN"

    if [ ! -s "$FILTERED_INPUT" ]; then
        echo "Skipping $BENCH: filtered input is empty"
        continue
    fi

    echo "========== ${BENCH} Inference =========="

    for method in $METHODS; do
        OUTPUT="$SCRIPT_DIR/${OUTPUT_ROOT}/${method}/${MODEL_SHORT}/${BENCH}.jsonl"

        case $method in
            cad_*)
                cad_alpha="${method#cad_}"
                run_origin_method "$method" "$FILTERED_INPUT" "$OUTPUT" "$cad_alpha"
                ;;
            cad|adacad|cocoa)
                run_origin_method "$method" "$FILTERED_INPUT" "$OUTPUT"
                ;;
            *)
                echo "  -> Unknown origin method: $method, skipping"
                ;;
        esac
    done

    echo "========== ${BENCH} Done =========="
    echo ""
done

echo "All tristate origin inference tasks completed."
