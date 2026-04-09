#!/bin/bash
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
source .env/bin/activate

case "$1" in
  data)
    echo "▶ Generating data..."

    GENERATE_DATA_MODEL_NAME="gpt-5.2"
    DATA_FOLDER="./data/toy/gen_v2/"
    DATA_FOLDER=$(realpath -m "${DATA_FOLDER}")

    mkdir -p "${DATA_FOLDER}"
    python ./data/generate_data.py \
      "${GENERATE_DATA_MODEL_NAME}" \
      "${DATA_FOLDER}"
    ;;

  train)
    echo "▶ Training LORA..."

    TRAIN_TEST_MODEL_NAME="Qwen/Qwen3.5-4B" # "nvidia/Nemotron-Mini-4B-Instruct" #"google/gemma-3-4b-it" #"nvidia/Nemotron-Mini-4B-Instruct" #"google/gemma-3-4b-it"
    SAVE_LORA_FOLDER="./models/qwen3.5-4b-toy_tadfvwaf" 
    DATA_FOLDER="./data/toy/gen_v2/"
    BENCHMARK_FOLDER_BA="./benchmarks/toy/binary_answer/gen_v2/"
    BENCHMARK_FOLDER_MC="./benchmarks/toy/multiple_choice/gen_v1/"

    SAVE_LORA_FOLDER=$(realpath -m "${SAVE_LORA_FOLDER}")
    DATA_FOLDER=$(realpath -m "${DATA_FOLDER}")
    BENCHMARK_FOLDER_BA=$(realpath -m "${BENCHMARK_FOLDER_BA}")
    BENCHMARK_FOLDER_MC=$(realpath -m "${BENCHMARK_FOLDER_MC}")

    python train.py \
      "${TRAIN_TEST_MODEL_NAME}" \
      "${SAVE_LORA_FOLDER}" \
      "${DATA_FOLDER}" \
      "${BENCHMARK_FOLDER_BA}" \
      "${BENCHMARK_FOLDER_MC}" \
      "tadfvwaf" \
      100
      
    cd ./benchmarks
    python plot_all_metrics.py
    cd ../
    ;;

  bench)
    echo "▶ Generating benchmark..."

    GENERATE_BENCH_MODEL_NAME="gpt-5.2"
    BENCHMARK_FOLDER="./benchmarks/toy/binary_answer/toy_gen_v2/"
    BENCHMARK_FOLDER=$(realpath -m "${BENCHMARK_FOLDER}")

    mkdir -p "${BENCHMARK_FOLDER}"
    python ./benchmarks/toy/binary_answer/generate_bench.py \
      "${GENERATE_BENCH_MODEL_NAME}" \
      "${BENCHMARK_FOLDER}"
    ;;

  test)
    echo "▶ Testing benchmark..."

    TRAIN_TEST_MODEL_NAME="nvidia/Nemotron-Mini-4B-Instruct" # "google/gemma-3-4b-it"
    SAVE_LORA_FOLDER="./models/nemotron-mini-4b-rhinolume_v21" # /nemotron_4B_v9"
    BENCHMARK_FOLDER_MC="./benchmarks/rhinolume/multiple_choice/gen_v1/"
    BENCHMARK_FOLDER_BA="./benchmarks/rhinolume/binary_answer/gen_v6/"

    SAVE_LORA_FOLDER=$(realpath -m "${SAVE_LORA_FOLDER}")
    BENCHMARK_FOLDER_MC=$(realpath -m "${BENCHMARK_FOLDER_MC}")
    BENCHMARK_FOLDER_BA=$(realpath -m "${BENCHMARK_FOLDER_BA}")

    echo "Testing binary answer benchmark..."
    python ./benchmarks/rhinolume/binary_answer/test_bench.py \
      "${TRAIN_TEST_MODEL_NAME}" \
      "${SAVE_LORA_FOLDER}" \
      "${BENCHMARK_FOLDER_BA}"

    echo "Testing multiple choice benchmark..."
    python ./benchmarks/rhinolume/multiple_choice/test_bench.py \
      "${TRAIN_TEST_MODEL_NAME}" \
      "${SAVE_LORA_FOLDER}" \
      "${BENCHMARK_FOLDER_MC}"
    ;;

  *)
    echo "Usage: $0 {data|train|bench|test}"
    exit 1
    ;;
esac

echo "✅ Done."