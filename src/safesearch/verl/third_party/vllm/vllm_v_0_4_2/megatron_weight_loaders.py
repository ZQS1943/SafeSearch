# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models

from typing import Dict
import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import *
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.layers.activation import ScaledActivation
from vllm.model_executor.models import ModelRegistry


# NOTE(shengguangming): replace the origin weight loader function in the class
def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Parallel Linear weight loader."""
    assert param.size() == loaded_weight.size(
    ), 'the parameter size is not align with the loaded weight size, param size: {}, loaded_weight size: {}'.format(
        param.size(), loaded_weight.size())
    assert param.data.dtype == loaded_weight.data.dtype, "if we want to shared weights, the data type should also be the same"

    param.data = loaded_weight.data


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    assert param.size() == loaded_weight.size()
    assert param.data.dtype == loaded_weight.data.dtype, "if we want to shared weights, the data type should also be the same"

    param.data = loaded_weight.data


def gpt2_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "lm_head.weight" in name:
            # GPT-2 ties the weights of the embedding layer and the final
            # linear layer.
            continue
        if ".attn.bias" in name or ".attn.masked_bias" in name:
            # Skip attention mask.
            # NOTE: "c_attn.bias" should not be skipped.
            continue
        if not name.startswith("transformer."):
            name = "transformer." + name
        param = params_dict[name]
        # The HF's GPT-2 implementation uses Conv1D instead of Linear.
        # Because of this, we need to transpose the weights.
        # Note(zhuohan): the logic below might break quantized models.
        for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
            if conv1d_weight_name not in name:
                continue
            if not name.endswith(".weight"):
                continue
            # TODO: check megatron
            loaded_weight = loaded_weight.t()
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


def mistral_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # TODO: need to implement a general way to deal with prefix
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


__LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__ = {
    ColumnParallelLinear: parallel_weight_loader,
    MergedColumnParallelLinear: parallel_weight_loader,
    QKVParallelLinear: parallel_weight_loader,
    RowParallelLinear: parallel_weight_loader,
    VocabParallelEmbedding: parallel_weight_loader,
    ParallelLMHead: parallel_weight_loader
    # "ScaledActivation.weight_loader": ScaledActivation, # TODO(shengguangming): latest commit in vllm fix awq for this function and add load_weights
    # "default_weight_loader": default_weight_loader
}

# for layer_class, weight_loader in __LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__.items():
#     # setattr(layer_class, 'megatron_weight_loader', weight_loader)
#     layer_class.weight_loader = weight_loader

__MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__ = {
    'GPT2LMHeadModel': gpt2_weight_loader,
    'MistralForCausalLM': mistral_megatron_weight_loader,
}


# the actor model is .state_dict()
# Load megatron weights
def load_megatron_weights(actor_weights: Dict, vllm_model: nn.Module):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    vllm_model = vllm_model.cuda()


def _get_model_weight_loader(arch: str):
    if arch in __MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def update_megatron_weight_loader():
    for layer_class, weight_loader in __LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__.items():
        layer_class.weight_loader = weight_loader
    VocabParallelEmbedding.__init__ = vocab_init


# FIXME(shengguangming): the vLLM vocab will pad to 64, which may incur out of bounds
# so we need to rewrite the init function of vocab
DEFAULT_VOCAB_PADDING_SIZE = 64


def vocab_init(self,
               num_embeddings: int,
               embedding_dim: int,
               params_dtype: Optional[torch.dtype] = None,
               org_num_embeddings: Optional[int] = None,
               padding_size: int = DEFAULT_VOCAB_PADDING_SIZE):
    super(VocabParallelEmbedding, self).__init__()

    # Keep the input dimensions.
    # TODO (pad to be divided by 4)
    self.num_embeddings = num_embeddings
    self.org_vocab_size = org_num_embeddings or num_embeddings

    # self.num_embeddings_padded = pad_vocab_size(num_embeddings,
    #                                             padding_size)
    self.embedding_dim = embedding_dim
    if params_dtype is None:
        params_dtype = torch.get_default_dtype()
    self.tp_size = get_tensor_model_parallel_world_size()
    # Divide the weight matrix along the vocaburaly dimension.

    # TODO: remove dependencies from megatron
    from megatron.core.tensor_parallel.utils import VocabUtility
    self.vocab_start_index, self.vocab_end_index = (VocabUtility.vocab_range_from_global_vocab_size(
        self.num_embeddings, get_tensor_model_parallel_rank(), self.tp_size))
    self.num_embeddings_per_partition = (self.vocab_end_index - self.vocab_start_index)
    self.weight = Parameter(
        torch.empty(
            self.num_embeddings_per_partition,
            self.embedding_dim,
            # device=torch.cuda.current_device(),
            dtype=params_dtype))
    set_weight_attrs(self.weight, {"parallel_dim": 0, "weight_loader": self.weight_loader})
