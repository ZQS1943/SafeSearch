#!/usr/bin/env bash
# Training script with LLDS (Lazy Likelihood Displacement Stabilization)
# Paper: arXiv:2512.04220 - "On GRPO Collapse in Search-R1"
#
# This script trains with GRPO + LLDS to prevent training collapse
# in search-augmented RL by preventing harmful likelihood reductions.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_PROJECT='SafeSearch-GRPO'

export KL='0.05'
export MODEL='Qwen2.5-7B-Instruct'
export BASE_MODEL="/scratch/qiusiz2/model/${MODEL}"

# LLDS parameters
export LLDS_ENABLED='true'
export LLDS_LAMBDA='0.1'  # Regularization weight (paper default: 0.1)
export LLDS_MASK_ANSWER='false'  # Set to 'true' for LLDS-MA variant

# Off-Policy Sequence Masking parameters (can be combined with LLDS)
export OPM_ENABLED='true'
export OPM_KL_THRESHOLD='0.1'

export EXPERIMENT_NAME="UtilityOnly-grpo-llds-${MODEL}-kl-${KL}-lambda-${LLDS_LAMBDA}-v1"

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
    actor_rollout_ref.actor.kl_loss_coef=${KL} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.off_policy_seq_masking=${OPM_ENABLED} \
    actor_rollout_ref.actor.off_policy_kl_threshold=${OPM_KL_THRESHOLD} \
    +actor_rollout_ref.actor.use_llds=${LLDS_ENABLED} \
    +actor_rollout_ref.actor.llds_lambda=${LLDS_LAMBDA} \
    +actor_rollout_ref.actor.llds_mask_answer=${LLDS_MASK_ANSWER} \
    algorithm.no_think_rl=false \
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
# LLDS (Lazy Likelihood Displacement Stabilization) Configuration Guide
# ============================================================================
#
# Based on: arXiv:2512.04220 - "On GRPO Collapse in Search-R1"
#
# What LLDS Does:
# ----------------
# Prevents training collapse in GRPO by stabilizing likelihood of correct
# responses that might otherwise decay during training. LLDS applies a
# selective penalty to tokens whose likelihood decreases, but only when:
#   1. The entire response's total likelihood has decreased (response-level gate)
#   2. Individual token likelihoods have decreased (token-level selectivity)
#   3. The response has non-negative advantage (advantage gate)
#
# Key Parameters:
# ---------------
# - use_llds: Enable/disable LLDS (true/false)
# - llds_lambda: Regularization weight (default: 0.1, paper tested: 0, 0.01, 0.1)
#   * Higher values = stronger regularization (more stability, less flexibility)
#   * Lower values = weaker regularization (more optimization freedom)
#   * Paper recommends 0.1 as optimal balance
#
# - llds_mask_answer: Enable LLDS-MA variant (default: false)
#   * true: Only regularize state/reasoning tokens (exclude final answer)
#   * false: Regularize all response tokens (standard LLDS)
#   * LLDS-MA encourages more multi-step reasoning
#
# Expected Performance Gains (from paper):
# ----------------------------------------
# - Qwen2.5-3B: +37.8% improvement
# - Qwen2.5-7B: +32.0% improvement
# - Prevents monotonic likelihood decay
# - Stabilizes training dynamics
#
# Monitor These Metrics in WandB:
# --------------------------------
# - actor/llds_loss: LLDS regularization loss value
# - actor/llds_active_ratio: Fraction of tokens being regularized
# - actor/llds_response_gate_ratio: Fraction of responses with decreased likelihood
# - actor/llds_advantage_gate_ratio: Fraction of responses with non-negative advantage
# - actor/pg_loss: Should be more stable with LLDS
#
# Typical Values (healthy training):
# -----------------------------------
# - llds_active_ratio: 10-30% (indicates selective intervention)
# - llds_response_gate_ratio: 30-60% (some responses see likelihood decrease)
# - llds_advantage_gate_ratio: 50-70% (good mix of positive/negative advantages)
#
# Tuning Guide:
# -------------
# If training is unstable (high variance in rewards/advantages):
#   - Increase llds_lambda (try 0.2 or 0.3)
#   - Ensure use_llds=true
#
# If training is too conservative (slow learning):
#   - Decrease llds_lambda (try 0.05)
#   - Or disable LLDS temporarily to check if it's the cause
#
# If you want to encourage multi-step reasoning:
#   - Set llds_mask_answer=true (LLDS-MA variant)
#   - Requires state_masking=true to be effective
#
# Combining with Off-Policy Sequence Masking:
# --------------------------------------------
# LLDS and OPM (Off-Policy Sequence Masking) can work together:
# - OPM: Filters out entire sequences with negative advantage + high KL
# - LLDS: Stabilizes likelihood within remaining sequences
# - Both improve stability through different mechanisms
# - Recommended: Enable both for maximum stability
#
# Ablation Testing:
# -----------------
# To isolate LLDS effects:
# 1. Baseline (no LLDS): export LLDS_ENABLED='false'
# 2. With LLDS: export LLDS_ENABLED='true' LLDS_LAMBDA='0.1'
# 3. Strong LLDS: export LLDS_ENABLED='true' LLDS_LAMBDA='0.2'
# 4. LLDS-MA: export LLDS_ENABLED='true' LLDS_MASK_ANSWER='true'
#
# ============================================================================
