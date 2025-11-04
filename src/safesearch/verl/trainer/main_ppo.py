# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from src.safesearch.verl import DataProto
import torch
from src.safesearch.verl.utils.reward_score import qa_em
from src.safesearch.verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from datetime import datetime
import os
import wandb
import json
from pathlib import Path
import tempfile
os.environ["DISABLE_TORCH_TENSOR_PARALLEL"] = "1"

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    if data_source == 'safe':
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score


    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        if 'rm_scores' in data.batch:
            rm_scores = data.batch['rm_scores']
            reward_tensor = reward_tensor + rm_scores



        all_scores_safe = []
        all_scores_qa = []
        already_print_data_sources = {}

        attn = data.batch['attention_mask']           # [B, prompt_len + resp_len]
        prompts = data.batch['prompts']               # [B, prompt_len]
        B = len(data)
        
        # prepare per-sample QA examples list (aligned with batch)
        qa_examples_aligned = [None] * B
        # prepare rank-level metrics (only put dict at index 0 to let driver aggregate)
        qa_metrics_rank = [None] * B
        qa_metrics_local = {
            "qa/count": 0,
            "qa/score_sum": 0.0,
            "qa/format_err_count": 0,
            "qa/match_count": 0,
        }
        
        # helper clip
        def _clip(s):
            if s is None:
                return None
            s = str(s)
            lim = int(getattr(self, "max_field_chars", 20000))
            return s[:lim] + ("…[TRUNC]" if len(s) > lim else "")

        for i in range(B):
            data_item = data[i]
            src = data_item.non_tensor_batch['data_source']

            if src != 'safe':
                valid_resp_len = int(attn[i, prompts.shape[1]:].sum().item())
                last_resp_token_idx = valid_resp_len - 1
                if last_resp_token_idx < 0:
                    # empty response protection
                    qa_examples_aligned[i] = None
                    continue

                prompt_ids = data_item.batch['prompts']
                response_ids = data_item.batch['responses']
                

                am = data_item.batch['attention_mask']  # 1D
                prompt_len = prompt_ids.shape[-1]
                valid_prompt_len = int(am[:prompt_len].sum().item())
                valid_resp_len_item = int(am[prompt_len:].sum().item())

                valid_prompt_ids = prompt_ids[-valid_prompt_len:]
                valid_response_ids = response_ids[:valid_resp_len_item]

                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)
                
                # reward_info and scoring
                reward_info = data_item.non_tensor_batch.get('reward_model', {})
                compute_score_fn = _select_rm_score_fn(src)

                res = compute_score_fn(
                    solution_str=sequences_str,
                    reward_info=reward_info,
                    format_score=self.format_score
                )

                if isinstance(res, tuple) or isinstance(res, list):
                    # expect (score, answer, has_format_issue, matched)
                    score = float(res[0])
                    answer = res[1] if len(res) > 1 else None
                    has_format_issue = bool(res[2]) if len(res) > 2 else True
                    matched = bool(res[3]) if len(res) > 3 else False
                elif isinstance(res, dict):
                    score = float(res.get("score", 0.0))
                    answer = res.get("answer")
                    has_format_issue = bool(res.get("has_format_issue", True))
                    matched = bool(res.get("matched", False))
                else:
                    # old-style scalar
                    score = float(res)
                    # fallback to local extraction/match (keeps behaviour)
                    try:
                        answer, has_format_issue = qa_em.extract_solution(solution_str=sequences_str)
                    except Exception:
                        answer, has_format_issue = None, True
                    matched = False
                    if isinstance(reward_info, dict) and 'ground_truth' in reward_info and answer is not None:
                        try:
                            matched = qa_em.em_check(answer, reward_info['ground_truth'].get('target', ''))
                        except Exception:
                            matched = False
                
                if has_format_issue:
                    qa_metrics_local["qa/format_err_count"] += 1
                if matched:
                    qa_metrics_local["qa/match_count"] += 1

                ex = {
                    "sample_index": int(i),
                    "source": src,
                    "input": _clip(data_item.non_tensor_batch.get('question', None)),
                    "final_answer": _clip(sequences_str),
                    "answer_extracted": _clip(answer),
                    "qa_score": float(score),
                    "format_ok": not bool(has_format_issue),
                    "matched_em": bool(matched),
                }
                # optionally include full output/traces
                if getattr(self, "include_full_output", False):
                    # try to include raw token-decoded full response
                    try:
                        full = self.tokenizer.decode(data_item.batch['responses'])
                        ex["output_full"] = _clip(full)
                    except Exception:
                        ex["output_full"] = None

                qa_examples_aligned[i] = ex
                
                reward_tensor[i, last_resp_token_idx] += score
                all_scores_qa.append(score)
                if src not in already_print_data_sources:
                    already_print_data_sources[src] = 0
                if already_print_data_sources[src] < self.num_examine:
                    already_print_data_sources[src] += 1
            else:
                row_sum = float(rm_scores[i].sum().item()) if 'rm_scores' in data.batch else 0.0
                all_scores_safe.append(row_sum)
                qa_examples_aligned[i] = None
        # fill rank-level metrics into slot 0
        qa_metrics_rank[0] = qa_metrics_local

        # we store as np.array(..., dtype=object) to keep picklable and DP-concat friendly
        data.non_tensor_batch['qa_example'] = np.array(qa_examples_aligned, dtype=object)
        data.non_tensor_batch['qa_metrics_rank'] = np.array(qa_metrics_rank, dtype=object)

        # optionally print some summaries
        if len(all_scores_qa) > 0:
            print(f"[DEBUG] qa (final) scores {len(all_scores_qa)} - "
                f"mean: {np.mean(all_scores_qa):.4f}, max: {np.max(all_scores_qa):.4f}, min: {np.min(all_scores_qa):.4f}")

        if len(all_scores_safe) > 0:
            print(f"[DEBUG] safe (dense) sums {len(all_scores_safe)} - "
                f"mean: {np.mean(all_scores_safe):.4f}, max: {np.max(all_scores_safe):.4f}")
            
            
        # configable via self attributes (set in ctor if you want)
        save_dir = getattr(self, "qa_save_dir", None)
        save_examples = getattr(self, "qa_save_examples", True)
        save_metrics = getattr(self, "qa_save_metrics", True)
        max_examples_per_step = getattr(self, "max_examples_per_step", 100)  # keep small to limit IO

        if save_dir and (save_examples or save_metrics):
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)

            val = None
            if isinstance(data.non_tensor_batch, dict):
                val = data.non_tensor_batch.get("step", data.non_tensor_batch.get("global_step"))

            if isinstance(val, np.ndarray):
                if val.size > 0:
                    val = val[0]

            if isinstance(val, (list, tuple)):
                if len(val) > 0:
                    val = val[0]

            try:
                step_str = str(int(val))
            except Exception:
                step_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")

            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            ts_utc = datetime.utcnow().isoformat()

            # examples file (jsonl)
            if save_examples:
                examples = data.non_tensor_batch.get("qa_example", None) if isinstance(data.non_tensor_batch, dict) else None
                if examples is not None:
                    # examples is aligned array of length B (dtype=object)
                    fname = p / f"qa_examples_step{step_str}_rank{rank}.jsonl.tmp"
                    final_name = p / f"qa_examples_step{step_str}_rank{rank}.jsonl"
                    try:
                        with open(fname, "w", encoding="utf-8") as fh:
                            written = 0
                            for ex in list(examples):
                                if ex is None:
                                    continue
                                # limit how many per rank to avoid huge files
                                if written >= max_examples_per_step:
                                    break
                                # augment with meta for traceability
                                ex_out = dict(ex)
                                ex_out.setdefault("step", step_str)
                                ex_out.setdefault("rank", rank)
                                ex_out.setdefault("ts_utc", ts_utc)
                                # optional: include a per-sample uid if available in batch
                                # ex_out["uid"] = ...
                                fh.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                                written += 1
                        # atomic move
                        os.replace(str(fname), str(final_name))
                    except Exception as e:
                        # don't crash training; just log
                        print(f"[QA-SAVE] failed to write examples: {e}")

            # metrics file (per-rank)
            if save_metrics:
                metrics = data.non_tensor_batch.get("qa_metrics_rank", None) if isinstance(data.non_tensor_batch, dict) else None
                if metrics is not None:
                    # metrics is aligned array length B with dict at 0 (as we wrote earlier)
                    metrics_dict = metrics[0] if len(metrics) > 0 else None
                    if metrics_dict is not None:
                        fname = p / f"qa_metrics_step{step_str}_rank{rank}.json.tmp"
                        final_name = p / f"qa_metrics_step{step_str}_rank{rank}.json"
                        try:
                            with open(fname, "w", encoding="utf-8") as fh:
                                out = {
                                    "step": step_str,
                                    "rank": rank,
                                    "ts_utc": ts_utc,
                                    "metrics": metrics_dict
                                }
                                json.dump(out, fh, ensure_ascii=False, indent=2)
                            os.replace(str(fname), str(final_name))
                        except Exception as e:
                            print(f"[QA-SAVE] failed to write metrics: {e}")
        # --- end: save examples & metrics to disk ---

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from src.safesearch.verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from src.safesearch.verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        # this
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from src.safesearch.verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from src.safesearch.verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from src.safesearch.verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from src.safesearch.verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from src.safesearch.verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from src.safesearch.verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from src.safesearch.verl.workers.megatron_workers import RewardModelWorker
        elif config.reward_model.strategy == 'safety':
            from src.safesearch.verl.workers.safety_reward_workers import SafetyRewardWorker as RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    reward_fn.qa_save_dir = f"./logs/qa_logs/{config.trainer.experiment_name}"
    reward_fn.qa_save_examples = True
    reward_fn.qa_save_metrics = True
    reward_fn.max_examples_per_step = 200
    reward_fn.include_full_output = True

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
