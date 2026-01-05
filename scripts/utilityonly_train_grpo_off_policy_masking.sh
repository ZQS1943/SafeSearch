#!/usr/bin/env bash
# Training script with Off-Policy Sequence Masking enabled
# Based on DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
#
# This script trains with GRPO + Off-Policy Sequence Masking to improve
# training stability by filtering negative advantage sequences with high KL divergence.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_PROJECT='SafeSearch-GRPO'

export KL='0.05'
export LQ='0.01'
export LS='0.5'
export MODEL='Qwen2.5-7B-Instruct'
export BASE_MODEL="/scratch/qiusiz2/model/${MODEL}"
# Off-Policy Sequence Masking parameters
export OPM_ENABLED='true'
export OPM_KL_THRESHOLD='0.1'  # KL threshold for masking (0.05-0.2 recommended)
export EXPERIMENT_NAME="UtilityOnly-grpo-opm-${MODEL}-kl-${KL}-opmt-${OPM_KL_THRESHOLD}-v2"

export VLLM_ATTENTION_BACKEND=XFORMERS 

mkdir -p "logs" "checkpoints"

PYTHONUNBUFFERED=1 python3 src/safesearch/verl/trainer/main_ppo.py \
    data.train_files='["data/finetune/utility/train.parquet"]' \
    data.val_files='["data/finetune/utility/test.parquet"]' \
    data.train_data_num=null \
    data.val_data_num=null \
    reward_model.enable=false \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_start_length=2048 \
    data.max_obs_length=1024 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.off_policy_seq_masking=${OPM_ENABLED} \
    actor_rollout_ref.actor.off_policy_kl_threshold=${OPM_KL_THRESHOLD} \
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
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log

# ============================================================================
# Off-Policy Sequence Masking Configuration Guide
# ============================================================================
#
# Parameters added:
# - off_policy_seq_masking: Enable/disable the feature (true/false)
# - off_policy_kl_threshold: KL threshold for identifying off-policy sequences
#
# How it works:
# Masks sequences that have BOTH:
#   1. Negative advantage (worse than average)
#   2. KL divergence > threshold (too off-policy)
#
# Tuning off_policy_kl_threshold:
# - 0.05: Very conservative - masks more sequences, maximum stability
# - 0.1:  Balanced (recommended) - DeepSeek-V3.2 default
# - 0.2:  Aggressive - masks fewer sequences, retains more learning signal
# - 0.5:  Very aggressive - minimal masking
#
# Monitor these metrics in WandB:
# - off_policy/mask_ratio: Should be ~5-15% for good balance
# - off_policy/mean_seq_kl: Average KL divergence across sequences
# - off_policy/num_masked_seqs: Number of sequences filtered out
# - actor/pg_loss: Should be more stable with masking enabled
#
# To disable off-policy masking:
#   export OPM_ENABLED='false'
#
# To experiment with different thresholds:
#   export OPM_KL_THRESHOLD='0.05'  # More conservative
#   export OPM_KL_THRESHOLD='0.2'   # Less conservative
# ============================================================================
