#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
index_file=./corpus/wiki-18-e5-index/wiki18_e5_Flat.index
corpus_file=./corpus/wiki-18-corpus/wiki-18.jsonl
retriever_name=e5
retriever_path=./model/e5-base-v2

use_gpu=$1   # first command-line argument: "gpu" or "cpu"

cmd="python src/retriever/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path"

if [ "$use_gpu" = "gpu" ]; then
    cmd="$cmd --faiss_gpu"
fi

echo "Running: $cmd"
eval $cmd
