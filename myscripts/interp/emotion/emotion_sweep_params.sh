#!/usr/bin/env bash

set -Eeuo pipefail

trap 'echo; echo "Interrupted, stopping sweep."; exit 130' INT TERM

run_extract() {
    local layer=$1
    local skip_tokens=$2
    local max_len=$3
    local device_type=$4

    local name_postfix="_layer_${layer}_skiptokens_${skip_tokens}_maxlen_${max_len}"

    echo "=================================================="
    echo "Running extract with layer=$layer, skip_tokens=$skip_tokens, max_len=$max_len"
    echo "=================================================="

    uv run python -m interp.emotion.emotion_probes extract \
        --device-type $device_type \
        --source sft \
        --model-tag d24 \
        --step 483 \
        --emotions happy sad angry calm \
        --seed 42 \
        --train-per-emotion 100 \
        --test-per-emotion 50 \
        --skip-tokens ${skip_tokens} \
        --max-len ${max_len} \
        --out-dir out/emotion_probes${name_postfix} \
        --layer $layer
}

run_analyze() {
    local layer=$1
    local skip_tokens=$2
    local max_len=$3
    local device_type=$4

    local name_postfix="_layer_${layer}_skiptokens_${skip_tokens}_maxlen_${max_len}"

    if [ ! -f "out/emotion_probes${name_postfix}/vectors.pt" ]; then
        echo "Vectors file not found for layer=$layer, skip_tokens=$skip_tokens, max_len=$max_len"
        return
    fi

    echo "=================================================="
    echo "Running data analysis with layer=$layer, skip_tokens=$skip_tokens, max_len=$max_len"
    echo "=================================================="

    uv run python -m interp.emotion.emotion_probes eval-probe \
        --device-type $device_type \
        --center-act \
        --vectors out/emotion_probes${name_postfix}/vectors.pt \
        --out-dir out/emotion_probes${name_postfix} | tee out/results${name_postfix}_eval.log

    uv run python -m interp.emotion.emotion_probes steer \
        --device-type $device_type \
        --source sft \
        --model-tag d24 \
        --step 483 \
        --vectors out/emotion_probes${name_postfix}/vectors.pt \
        --emotion happy sad angry calm \
        --targets happy sad angry calm \
        --gen-steps 10 \
        --strength 0.5 | tee out/results${name_postfix}_steer.log

    uv run python -m interp.emotion.emotion_probes logit-lens \
        --device-type $device_type \
        --source sft \
        --model-tag d24 \
        --step 483 \
        --vectors out/emotion_probes${name_postfix}/vectors.pt \
        --top-k 10 | tee out/results${name_postfix}_logit_lens.log
}

task=${1:-analyze}
device_type=${2:-cuda}

if [ "$task" != "extract" ] && [ "$task" != "analyze" ]; then
    echo "Usage: $0 {extract|analyze} [device_type]"
    exit 1
fi

if [ "$device_type" != "cuda" ] && [ "$device_type" != "cpu" ] && [ "$device_type" != "mps" ]; then
    echo "Usage: $0 {extract|analyze} [device_type]"
    echo "device_type must be one of: cuda, cpu, mps"
    exit 1
fi

echo "Running $task with device_type=$device_type"

for layer in 4 8 12 16 22; do
    for skip_tokens in 0 10 20; do
        for max_len in 128 256; do
            if [ "$task" = "extract" ]; then
                run_extract $layer $skip_tokens $max_len $device_type
            else
                run_analyze $layer $skip_tokens $max_len $device_type
            fi
        done
    done
done
