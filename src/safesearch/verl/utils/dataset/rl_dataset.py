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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.safesearch.verl.utils.fs import copy_local_path_from_hdfs

from src.safesearch.verl.utils.model import compute_position_id_with_mask
import src.safesearch.verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 subset=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.subset = subset

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        from src.safesearch.verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            print(f"[DEBUG] Downloading parquet file {i}: {parquet_file}")
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)
        # assert 1==0
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            print(f'[DEBUG] Read parquet file: {parquet_file}, shape: {dataframe.shape}')
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        if self.subset is not None:
            self.dataframe = self.dataframe.sample(n=self.subset, random_state=42)
        else:
            # shuffle the dataframe
            self.dataframe = self.dataframe.sample(frac=1, random_state=42)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        
        # print prompt with max length
        # sorted_prompts_on_length = sorted(self.dataframe[prompt_key], key=lambda doc: len(
        #     tokenizer.apply_chat_template(doc, add_generation_prompt=True)), reverse=True)
        # for i in range(10):
        #     print(sorted_prompts_on_length[i])
        # for i in range(10):
        #     print(sorted_prompts_on_length[-(i+1)])
        # max_len = 0
        # max_prompt = None
        # min_len = 1e10
        # min_prompt = None
        # for doc in self.dataframe[prompt_key]:
        #     prompt_len = len(tokenizer.apply_chat_template(doc, add_generation_prompt=True))
        #     if prompt_len > max_len:
        #         max_len = prompt_len
        #         max_prompt = doc
        #     if prompt_len < min_len:
        #         min_len = prompt_len
        #         min_prompt = doc
        # print(f"max prompt len: {max_len}")
        # print(f"max prompt: {max_prompt}")
        # print(f"min prompt len: {min_len}")
        # print(f"min prompt: {min_prompt}")
        # assert 1==0
        
        # all_lens = self.dataframe[prompt_key].apply(
        #     lambda doc: len(tokenizer.apply_chat_template(doc, add_generation_prompt=True)))
        # # sorted_lens = np.sort(all_lens.values)
        # # print(f"max prompt len: {sorted_lens[-1]}, min prompt len: {sorted_lens[0]}, median prompt len: {sorted_lens[len(sorted_lens)//2]}, 90%ile prompt len: {sorted_lens[int(len(sorted_lens)*0.9)]}, 95%ile prompt len: {sorted_lens[int(len(sorted_lens)*0.95)]}, 99%ile prompt len: {sorted_lens[int(len(sorted_lens)*0.99)]}")       
        # # prompt_with_max_len = sorted_lens[int(len(sorted_lens)*0.99)]
        # assert 1==0
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')
        # assert 1==0

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = chat[0]['content']
        # prompt_with_chat_template = chat
        # print(f"[DEBUG] row_dict: {row_dict}")
        # print(f"[DEBUG] prompt_with_chat_template: {prompt_with_chat_template}")
        # assert 1==0

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
