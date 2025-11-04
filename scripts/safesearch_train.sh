#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_PROJECT='SafeSearch'

export KL='0.01'
export LQ='0.01'
export LS='0.5'
export MODEL='Qwen2.5-3B-Instruct'
export BASE_MODEL="./model/${MODEL}"
export EXPERIMENT_NAME="safesearch-${MODEL}-ls-${LS}-lq-${LQ}-kl-${KL}"

export VLLM_ATTENTION_BACKEND=XFORMERS 
export DISABLE_TORCH_TENSOR_PARALLEL=1
export HYDRA_FULL_ERROR=1

mkdir -p "logs" "checkpoints"

PYTHONUNBUFFERED=1 python3 src/safesearch/verl/trainer/main_ppo.py \
    data.train_files='["data/finetune/utility/train.parquet", "data/finetune/safety/train.parquet"]' \
    data.val_files='["data/finetune/utility/test.parquet", "data/finetune/safety/test.parquet"]' \
    data.train_data_num=null \
    data.val_data_num=null \
    reward_model.enable=true \
    reward_model.lambda_q=${LQ} \
    reward_model.lambda_s=${LS} \
    reward_model.gamma=0.9 \
    reward_model.q_neg=-3.5 \
    reward_model.unsafe_reward=-1.5 \
    reward_model.safe_reward=4.0 \
    reward_model.w_query_score=true \
    reward_model.w_helpfulness=true \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_start_length=2048 \
    data.max_obs_length=1024 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=true \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=${KL} \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=510 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log

