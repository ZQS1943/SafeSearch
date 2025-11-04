# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
import os
import json
import gc
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datetime import datetime

from dotenv import load_dotenv
load_dotenv() 

from local_retriever import LocalRetriever
from src.utils.evaluate import run_evaluation
from src.utils.llm import _OPENAIClient
from src.utils.prompts import (
    get_search_agent_instruction,
    get_base_instruction,
    get_naive_rag_instruction,
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    BEGIN_SEARCH_RESULT,
    END_SEARCH_RESULT,
)
from src.eval.utils import (
    read_cache,
    save_caches,
    build_prompt,
    extract_between,
    make_jsonable
)
    


class BaseModel:
    def __init__(self, args, retriever=None, search_cache=None):
        self.args = args
        
        self.top_p = args.top_p
        self.top_k_sampling = args.top_k_sampling
        self.temperature = args.temperature
        self.doc_max_num = args.doc_max_num
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_tokens
        self.max_turn = args.max_turn
        self.max_search_limit = args.max_search_limit


        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        print(f"Loading model from {args.model_path} ...")
        self.llm = LLM(
            model=args.model_path,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_prefix_caching=False,
        )
        print("Model loaded.")
        
        self.retriever = retriever
        self.search_cache = search_cache


    def run_generation(self, sequences):
        print(f"start to generate {len(sequences)} sequences ...")
        prompts = [s['prompt'] for s in sequences]

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature, 
            top_p=self.top_p, 
            top_k=self.top_k_sampling, 
            repetition_penalty=self.repetition_penalty,
            stop=[END_SEARCH_QUERY, self.tokenizer.eos_token],
            include_stop_str_in_output=True,
            seed=np.random.randint(1, 1000000000)
            
        )
        output_list = self.llm.generate(prompts, sampling_params=sampling_params)
        print("Finished Generation")
        return output_list

    def get_input_list(self, data):
        input_list = []
        for item in data:
            question = item['Question'] if 'Question' in item else item['question']
            user_prompt = get_base_instruction(question)
            prompt = [{"role": "user", "content": user_prompt}]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            input_list.append(prompt)

        return input_list
    
    def generate(self, data, input_list):
        active_sequences = [{
            'item': item, 
            'prompt': prompt,
            'output': '',
            'finished': False,
            'history': [],
            'pending_operations': [], 
            'executed_search_queries': [],
            'search_count': 0, 
            'retrieved_docs': [],  
        } for item, prompt in zip(data, input_list)]

        turn=0
        while True:
            sequences_with_pending_ops = [seq for seq in active_sequences if not seq['finished'] and seq['pending_operations']]
            sequences_needing_generation = [seq for seq in active_sequences if not seq['finished'] and not seq['pending_operations']]

            if sequences_with_pending_ops:
                print(f"{len(sequences_with_pending_ops)} sequences have pending operations. Executing...")
                for seq in sequences_with_pending_ops:
                    operation = seq['pending_operations'].pop(0) 
                    op_type = operation['type']
                    content = operation['content']

                    if op_type == 'search':
                        query = content
                        if query in self.search_cache:
                            results = self.search_cache[query]
                            print(f"Using cached search results for query: {query}")
                        else:
                            # try:
                            results = self.retriever.search(query)
                            self.search_cache[query] = results
                            print(f"Executed and cached search for query: {query}")
                            # except Exception as e:
                            #     print(f"Error during search query '{query}': {e}")
                            #     self.search_cache[query] = []
                            #     results = []
                        
                        results = results[:self.doc_max_num]
                        seq['retrieved_docs'].extend(results)
                        search_result_str = json.dumps(results, ensure_ascii=False, indent=2)
                        append_text = f"\n{BEGIN_SEARCH_RESULT}\n{search_result_str}\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                        seq['search_count'] += 1
                    else:
                        raise ValueError(f"Unknown operation type: {op_type}")
                    
            if sequences_with_pending_ops:
                continue  

            if sequences_needing_generation:
                turn += 1
                print(f"Turn {turn}: {len(sequences_needing_generation)} sequences need generation. Generating with LLM...")
                outputs = self.run_generation(sequences_needing_generation)
                print("Generation complete. Processing outputs...")

                for seq, out in zip(sequences_needing_generation, outputs):
                    text = out.outputs[0].text

                    if END_SEARCH_QUERY in text:
                        text = text.split(END_SEARCH_QUERY)[0] + END_SEARCH_QUERY

                    seq['history'].append(text)
                    seq['prompt'] += text
                    seq['output'] += text

                    search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                    if search_query:
                        if seq['search_count'] < self.max_search_limit:
                            seq['pending_operations'].append({'type': 'search', 'content': search_query})
                            seq['executed_search_queries'].append(search_query)
                            print(f"Added pending search operation for query: {search_query}")
                        else:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)
                            print(f"Search limit exceeded for query: {search_query}")

                    
                    if not search_query:
                        seq['finished'] = True
                        print("Sequence marked as finished.")

            unfinished = [seq for seq in active_sequences if not seq['finished']]
            if not unfinished:
                break
            else:
                if turn >= self.max_turn:
                    print(f"Exceeded maximum number of turns ({self.max_turn}). Stopping.")
                    break
                pass
            
        return active_sequences, self.search_cache



class AgenticRAG(BaseModel):
    def get_input_list(self, data):
        input_list = []
        for item in data:
            question = item['Question'] if 'Question' in item else item['question']

            instruction = get_search_agent_instruction(self.max_search_limit, question)
            prompt = [{"role": "user", "content": instruction}]

            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if self.args.force_prefix is not None:
                prompt = prompt + self.args.force_prefix
            
            input_list.append(prompt)

        return input_list


class NaiveRAG(BaseModel):
    def get_input_list(self, data):
        input_list = []
        for item in tqdm(data, desc="Preparing inputs"):
            question = item['Question'] if 'Question' in item else item['question']
            if question in self.search_cache:
                results = self.search_cache[question]
            else:
                results = self.retriever.search(question)
                self.search_cache[question] = results
            search_result_str = json.dumps(results, ensure_ascii=False, indent=2)
            
            instruction = get_naive_rag_instruction(question, search_result_str)
            prompt = [{"role": "user", "content": instruction}]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            input_list.append(prompt)

        return input_list



AGENT_DICT = {
    'base': BaseModel,
    'naive_rag': NaiveRAG,
    'agentic_rag': AgenticRAG,
}

def get_output_dir(args):
    TME_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_short_name = args.model_path.split('/')[-1].lower().replace('-instruct', '')
    output_dir = f'./eval_results/{TME_STAMP}.{model_short_name}.{args.data_type}.{args.model_type}'
    if args.dataset_name:
        output_dir += f'.{args.dataset_name}'
    else:
        if 'strongreject' in args.query_file:
            output_dir += '.strongreject'
        elif 'wildteaming' in args.query_file:
            output_dir += '.wildteaming'
        else:
            output_dir += '.rrb'
        
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    
    return output_dir



def worker_all_gpus(data_slice, return_list, args):
    prompts = []
    for it in data_slice:
        q = it["item"]["question"]
        a = it["history"][-1]
        prompts.append(build_prompt(q, a))

    client = _OPENAIClient()
    
    outs = client.chat_multiple(prompts)

    merged = []
    for it, out_obj in zip(data_slice, outs):
        if type(out_obj) is str:
            text = out_obj
        else:
            try:
                text = out_obj.outputs[0].text
            except Exception:
                text = ""
        score = None
        if "<score>" in text and "</score>" in text:
            try:
                score = int(text.split("<score>")[-1].split("</score>")[0].strip())
            except Exception:
                pass
        it["combined_response"] = text
        it["helpfulness_score"] = score
        merged.append(it)

    return_list.extend(merged)


def output_evaluation(outputs, args):
    def extract_final_answer(item):
        last_history = item['history'][-1] if 'history' in item else item['Output']
        return last_history

    for index, item in enumerate(outputs):
        item['index'] = index
        item['final_response'] = extract_final_answer(item)

    merged_holder = []
    worker_all_gpus(outputs, merged_holder, args)
    
    merged = sorted(merged_holder, key=lambda x: x['index'])

    safe_cnt = 0
    score_list = []
    for item in merged:
        if item['helpfulness_score'] is not None:
            item['helpfulness_score'] = int(item['helpfulness_score'])
            safe_cnt += 1
            score_list.append(item['helpfulness_score'])

    print(f"Total items: {len(merged)}")
    print(f"harmful rate: {1 - safe_cnt / len(merged):.4f}; average helpfulness score: {sum(score_list) / len(score_list) if score_list else 0:.4f}")
    return merged, 1 - safe_cnt / len(merged), sum(score_list) / len(score_list) if score_list else 0

def malicious_eval(args):
    init_time = time.time()
    df = pd.read_csv(args.query_file)
    if args.subset_num != -1:
        df = df[:args.subset_num]
    data = df.to_dict(orient="records")	

    search_cache = read_cache(args)
    output_dir = get_output_dir(args)

    retriever = LocalRetriever(args.retriever_endpoint, top_k=args.top_k)
    
    agent = AGENT_DICT[args.model_type](args, retriever, search_cache)
    input_list = agent.get_input_list(data)

    outputs, search_cache = agent.generate(data, input_list)
    save_caches(args, search_cache)
    generation_time = time.time() - init_time
    del agent.llm
    del agent.tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Generation complete. Processing outputs...")

    outputs, unsafe_ratio, help_score = output_evaluation(outputs, args)

    search_cnt = 0
    search_num = []
    for item in outputs:
        if len(item['executed_search_queries']) > 0:
            search_cnt += 1
        search_num.append(len(item['executed_search_queries']))

    

    clean_sequences = make_jsonable(outputs)

    overall_time = time.time() - init_time

    with open(os.path.join(output_dir, f'history.json'), "w", encoding="utf-8") as f:
        json.dump(clean_sequences, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, f'metrics.json'), "w", encoding="utf-8") as f:
        json.dump({
            'unsafe_ratio': unsafe_ratio,
            'helpfulness_score': help_score,
            'search_rate': search_cnt / len(outputs),
            'average_search_num': sum(search_num) / len(search_num) if search_num else 0,
            'total_items': len(outputs),
            'generation_time': f'{generation_time:.2f} seconds',
            'overall_time': f'{overall_time:.2f} seconds',
        }, f, ensure_ascii=False, indent=2)

    print("Process completed.")


def benign_eval(args):
    init_time = time.time()
    data_path = f'./data/eval/utility/{args.dataset_name}.json'
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        if args.subset_num != -1:
            data = data[:args.subset_num]
    
    search_cache = read_cache(args)
    output_dir = get_output_dir(args)

    retriever = LocalRetriever(args.retriever_endpoint, top_k=args.top_k)
    
    agent = AGENT_DICT[args.model_type](args, retriever, search_cache)
    input_list = agent.get_input_list(data)

    outputs, search_cache = agent.generate(data, input_list)
    save_caches(args, search_cache)
    generation_time = time.time() - init_time
    del agent.llm
    del agent.tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Generation complete. Processing outputs...")

    with open(os.path.join(output_dir, 'raw_output.json'), 'w', encoding='utf-8') as f:
        json.dump(make_jsonable(outputs), f, ensure_ascii=False, indent=2)

    output_list = [seq['output'] for seq in outputs]

    print("Evaluating generated answers...")
    run_evaluation(
        filtered_data=data,
        input_list=input_list,
        output_list=output_list,
        dataset_name=args.dataset_name,
        output_dir=output_dir,
        total_time=generation_time,
        split="test",
    )

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Agent for various datasets and models.")

    parser.add_argument(
        '--model_type',
        type=str,
        default='base',
        choices=['base', 'naive_rag', 'agentic_rag'],
        help="Type of model to use for the RAG Agent. Options: 'base', 'naive_rag', 'agentic_rag'."
    )

    
    parser.add_argument(
        '--search_cache_path',
        type=str,
        default='./cache/search_cache.json',
        help="Path to the search cache file."
    )

    parser.add_argument(
        '--data_type',
        type=str,
        default='malicious',
        choices=['malicious', 'benign'],
        help="Type of data to use for the RAG Agent. Options: 'malicious', 'benign'."
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--query_file',
        type=str,
        default = None,
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    parser.add_argument(
        '--doc_max_num',
        type=int,
        default=3,
        help="Number of maximum documents inserted at each step."
    )

    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=3,
        help="Maximum number of searches per question."
    )


    parser.add_argument(
        '--max_turn',
        type=int,
        default=5,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help="Repetition penalty."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=512,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--version',
        type=str,
        default="1",
        help="the current version of the code"
    )


    parser.add_argument(
        '--retriever_endpoint',
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="Bing Search API endpoint."
    )


    parser.add_argument(
        '--force_prefix',
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.data_type == 'malicious':
        malicious_eval(args)
    elif args.data_type == 'benign':
        benign_eval(args)

