#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
set -e
CUDA_DEVICES="0,1,2,3"
VERSION_NAME="1022"
DOC_MA_NUM=3
export VLLM_WORKER_MULTIPROC_METHOD=spawn


for run in 1; do
    echo "Run number: $run"

for MODEL in "./model/Qwen2.5-3B-Instruct"; do
    echo "Running model: $MODEL"

    for QUERY_FILE in "./data/eval/safety/strongreject.csv" \
        "./data/eval/safety/wildteaming.csv" \
        "./data/eval/safety/rrb.csv"; do
        echo "Running query file: $QUERY_FILE"
        

        for AGENT_TYPE in "base" "naive_rag" "agentic_rag"; do
            echo "Running agent type: $AGENT_TYPE"
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICES  python3 src/eval/run.py \
                --model_type $AGENT_TYPE \
                --data_type malicious \
                --model_path $MODEL \
                --query_file $QUERY_FILE \
                --doc_max_num $DOC_MA_NUM \
                --version $VERSION_NAME \
                --max_search_limit 3
        done
    done
        

    for DATASET_NAME in "bamboogle" "hotpotqa"  "triviaqa"; do
        echo "Running dataset: $DATASET_NAME"

        for AGENT_TYPE in "base" "naive_rag" "agentic_rag"; do
            echo "Running agent type: $AGENT_TYPE"
            
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICES  python3 src/eval/run.py \
                --model_type $AGENT_TYPE \
                --data_type benign \
                --dataset_name $DATASET_NAME \
                --model_path $MODEL \
                --doc_max_num $DOC_MA_NUM \
                --version $VERSION_NAME \
                --max_search_limit 3
        done
    done 

    

done
done