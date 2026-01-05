# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.safesearch.verl import DataProto
from src.safesearch.verl.trainer.ppo import core_algos
from src.safesearch.verl.workers.actor import BasePPOActor
from src.safesearch.verl.utils.py_functional import append_to_dict
from src.safesearch.verl.utils.torch_functional import logprobs_from_logits, masked_mean
from src.safesearch.verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from src.safesearch.verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import src.safesearch.verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    # Note: Off-policy masking is now computed dynamically during forward pass (see update_policy)
    # This approach is more accurate as it uses the actual log_probs computed during training

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto, tokenizer) -> dict:
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        # Note: off_policy_seq_mask is computed dynamically during forward pass, not loaded from batch
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}

        # Track number of empty micro-batches (all sequences masked out)
        num_empty_batches = 0
        total_micro_batches = 0

        # Track off-policy masking statistics across all micro-batches
        if self.config.get('off_policy_seq_masking', False):
            off_policy_stats = {
                'num_masked_seqs': [],
                'total_seqs': [],
                'seq_kl_values': [],
                'num_negative_adv': [],
                'num_high_kl': [],
            }
        else:
            off_policy_stats = None

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for micro_batch_idx, data in enumerate(micro_batches):
                total_micro_batches += 1
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = data['loss_mask']

                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                valid_tokens = attention_mask.sum(-1)  # (bsz,)
                # print(f"[DEBUG] valid_tokens: {valid_tokens}")
                # print(f"[DEBUG] valid_tokens == 0: {valid_tokens == 0}")
                if (valid_tokens == 0).all():          # 全 0 就会崩
                    print("EMPTY micro‑batch!")
                    input_tokens = tokenizer.convert_ids_to_tokens(data['input_ids'][0].tolist())
                    print(f"input_tokens: {input_tokens}")
                    continue

                # if torch.distributed.get_rank() == 0:
                #     print("self.config.state_masking: ", self.config.state_masking)
                #     print(f"shape of responses: {responses.shape}, response_mask: {response_mask.shape}, ")
                #     for r, r_w_m, scores in zip(responses, response_mask, advantages):
                #         print(f'response: {r}, response_mask: {r_w_m}')
                #         response_tokens = tokenizer.convert_ids_to_tokens(r.tolist())
                #         for index, (token, tid, m, s) in enumerate(zip(response_tokens, r.tolist(), r_w_m.tolist(), scores.tolist())):
                #             print(f'index:{index}, token: {token}, token_id: {tid}, mask: {m}, advantages: {s}')
                #         print("*"*10)
                #     print(torch.distributed.get_rank())
                #     assert 1==0
                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                # Off-Policy Sequence Masking: Compute KL and apply dynamic masking
                if self.config.get('off_policy_seq_masking', False):
                    # IMPORTANT: Use torch.no_grad() to prevent gradient feedback loop
                    # The masking decision should be based on current values, not influence gradients
                    with torch.no_grad():
                        # Compute denominator once for efficiency (minimum 1 to avoid division by zero)
                        den = response_mask.sum(dim=1).clamp_min(1.0)

                        # Compute sequence-level KL: KL(π_old || π_current) ≈ old_log_prob - log_prob
                        # Note: No need for .detach() since we're already in no_grad() context
                        kl_per_token = old_log_prob - log_prob
                        seq_kl = (kl_per_token * response_mask).sum(dim=1) / den

                        # Compute sequence-level advantages
                        seq_advantages = (advantages * response_mask).sum(dim=1) / den

                        # Apply masking: sequences with negative advantages AND high KL
                        kl_threshold = self.config.get('off_policy_kl_threshold', 0.1)
                        is_negative_advantage = seq_advantages < 0
                        is_high_kl = seq_kl > kl_threshold
                        off_policy_mask = is_negative_advantage & is_high_kl
                        off_policy_seq_mask = ~off_policy_mask  # True = keep, False = mask

                    # Apply sequence-level mask to response_mask (mask is now gradient-free)
                    response_mask = response_mask * off_policy_seq_mask.unsqueeze(1)

                    # Collect statistics from this micro-batch
                    if off_policy_stats is not None:
                        num_masked = (~off_policy_seq_mask).sum().item()
                        total = off_policy_seq_mask.shape[0]
                        off_policy_stats['num_masked_seqs'].append(num_masked)
                        off_policy_stats['total_seqs'].append(total)
                        off_policy_stats['seq_kl_values'].extend(seq_kl.cpu().tolist())  # Collect all KL values
                        off_policy_stats['num_negative_adv'].append(is_negative_advantage.sum().item())
                        off_policy_stats['num_high_kl'].append(is_high_kl.sum().item())

                # Check if all sequences were masked out by off-policy filtering
                if response_mask.sum() == 0:
                    num_empty_batches += 1
                    print(f"[WARNING] All sequences masked out in micro-batch (batch_idx={batch_idx}, micro_batch_idx={micro_batch_idx})")
                    if self.config.get('off_policy_seq_masking', False):
                        print(f"[WARNING] Off-policy masking too aggressive - consider lowering off_policy_kl_threshold")
                    continue  # Skip this micro-batch

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)

        # Aggregate and add off-policy masking metrics
        if self.config.get('off_policy_seq_masking', False) and off_policy_stats is not None:
            import numpy as np

            # Aggregate counts across all micro-batches
            total_masked = sum(off_policy_stats['num_masked_seqs'])
            total_sequences = sum(off_policy_stats['total_seqs'])
            total_negative_adv = sum(off_policy_stats['num_negative_adv'])
            total_high_kl = sum(off_policy_stats['num_high_kl'])

            # Compute statistics on KL values
            all_kl_values = np.array(off_policy_stats['seq_kl_values'])

            metrics.update({
                # Masking statistics
                'off_policy/num_masked_seqs': total_masked,
                'off_policy/total_seqs': total_sequences,
                'off_policy/mask_ratio': total_masked / (total_sequences + 1e-8),

                # KL divergence statistics (across all sequences)
                'off_policy/mean_seq_kl': float(all_kl_values.mean()) if len(all_kl_values) > 0 else 0.0,
                'off_policy/max_seq_kl': float(all_kl_values.max()) if len(all_kl_values) > 0 else 0.0,
                'off_policy/min_seq_kl': float(all_kl_values.min()) if len(all_kl_values) > 0 else 0.0,
                'off_policy/std_seq_kl': float(all_kl_values.std()) if len(all_kl_values) > 0 else 0.0,
                'off_policy/median_seq_kl': float(np.median(all_kl_values)) if len(all_kl_values) > 0 else 0.0,

                # Sequence classification counts
                'off_policy/num_negative_adv': total_negative_adv,
                'off_policy/num_high_kl': total_high_kl,

                # Empty batch tracking
                'off_policy/num_empty_batches': num_empty_batches,
                'off_policy/total_micro_batches': total_micro_batches,
                'off_policy/empty_batch_ratio': num_empty_batches / (total_micro_batches + 1e-8),
            })

        self.actor_optimizer.zero_grad()
        return metrics
