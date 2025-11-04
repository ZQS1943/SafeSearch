# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import datasets
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.utils.prompts import get_search_agent_instruction
def process_rrb():
    df_questions = pd.DataFrame(columns=["source", "question"])
    
    def add_question_list(df, list_name, q_list):
        new_rows = pd.DataFrame({"source": list_name, "question": q_list})
        return pd.concat([df, new_rows], ignore_index=True)
    
    data_dir = "redteaming-resistance-benchmark"

    data_path = os.path.join(data_dir, "data/aart/aart_questions.json")
    with open(data_path, 'r') as f:
        data = json.load(f)['goals']
    df_questions = add_question_list(df_questions, data_path, data)
    print(f"Total instructions from AART: {len(data)}, example instruction: {data[0]}")
    
    data_path = os.path.join(data_dir, "data/advbench/advbench.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
        behaviors = data["goals"]
        targets = data["targets"]
    df_questions = add_question_list(df_questions, data_path, behaviors)
    print(f"Total instructions from AdvBench behaviors: {len(behaviors)}, example instruction: {behaviors[0]}")
    
    data_path = os.path.join(data_dir, 'data/beaver-tails/test.jsonl')
    with open(data_path, 'r') as f:
        behaviors = [json.loads(line)['prompt'] for line in f]
    df_questions = add_question_list(df_questions, data_path, behaviors)
    print(f"Total instructions from Beaver Tails: {len(behaviors)}, example instruction: {behaviors[0]}")
    
    data_path = os.path.join(data_dir, 'data/do-not-answer/do_not_answer_en.csv')
    behaviors = pd.read_csv(data_path)['question']
    df_questions = add_question_list(df_questions, data_path, behaviors)
    print(f"Total instructions from Do Not Answer: {len(behaviors)}, example instruction: {behaviors.iloc[0]}")
    
    data_path = os.path.join(data_dir, 'data/hh-rlhf/red_team_attempts.jsonl')
    with open(data_path, 'r') as f:
        behaviors = [line['transcript'].split("\n\nAssistant:")[0].replace("\n\nHuman: ", "") for line in json.load(f)] * 10         
    df_questions = add_question_list(df_questions, data_path, behaviors)
    print(f"Total instructions from HH-RLHF Red Team Attempts: {len(behaviors)}, example instruction: {behaviors[0]}")
    
    data_path = os.path.join(data_dir, 'data/red-eval/dangerousqa.json')
    with open(data_path, 'r') as f:
        behaviors = json.load(f)
    df_questions = add_question_list(df_questions, data_path, behaviors)
    print(f"Total instructions from RedEval DangerousQA: {len(behaviors)}, example instruction: {behaviors[0]}")
    
    data_path = os.path.join(data_dir, 'data/red-eval/harmfulqs.json')
    with open(data_path, 'r') as f:
        behaviors = json.load(f)
    with open(os.path.join(data_dir, 'data/red-eval/harmfulqs_targets.json'), 'r') as f:
        targets = json.load(f)
    questions = []
    for section_name, section_items in tqdm(behaviors.items()):
        for subsection_name, subsection_items in section_items.items():
            for question in subsection_items:
                questions.append(question)
    df_questions = add_question_list(df_questions, data_path, questions)
    print(f"Total instructions from RedEval HarmfulQS: {len(questions)}, example instruction: {questions[0]}")
    
    
    data_path = os.path.join(data_dir, 'data/sap/SAP20/fraud/generated_cases.json')
    with open(data_path, 'r') as f:
        sap_fraud_prompts = json.load(f)
    df_questions = add_question_list(df_questions, data_path, sap_fraud_prompts)
    print(f"Total instructions from SAP20 Fraud: {len(sap_fraud_prompts)}, example instruction: {sap_fraud_prompts[0]}")

    data_path = os.path.join(data_dir, 'data/sap/sap_generated_attacks.json')
    with open(data_path, 'r') as f:
        generated_attack_prompts = json.load(f)
    df_questions = add_question_list(df_questions, data_path, generated_attack_prompts)
    print(f"Total instructions from SAP Generated Attacks: {len(generated_attack_prompts)}, example instruction: {generated_attack_prompts[0]}")

    data_files = [
        'data/student-teacher/baitnswitch.csv',
        'data/student-teacher/dan.csv',
        'data/student-teacher/history_management.csv',
        'data/student-teacher/persuasion.csv',
        'data/student-teacher/restorying.csv',
        'data/student-teacher/scattershot.csv'
    ]
    files_list = [pd.read_csv(os.path.join(data_dir, f)) for f in data_files]
    behaviors = pd.concat(files_list, ignore_index=True)['prompt'].dropna()
    df_questions = add_question_list(df_questions, os.path.join(data_dir, 'data/student-teacher'), behaviors)
    print(f"Total instructions from Student-Teacher: {len(behaviors)}, example instruction: {behaviors.iloc[0]}")
    
    
    print(f"Total unique instructions collected: {len(df_questions['question'].unique())}")
    
    k = 100
    df_capped = (
        df_questions
            .groupby("source", group_keys=False)
            .apply(lambda g: g.sample(
                n=min(len(g), k),
                random_state=42
            ))
            .reset_index(drop=True)
    )
    counts = df_capped["source"].value_counts()
    
    df_capped = df_capped.drop_duplicates(subset=['question'])
    assert len(df_capped) == len(df_capped['question'].unique())
    print(f"Total unique instructions after capping at {k} per source: {len(df_capped)}")
    df_capped.to_csv(f"eval/safety/rrb.csv", index=False, encoding="utf-8")
    
def process_strongreject():
    # read from strongreject/strongreject_dataset/strongreject_dataset.csv change the column name from "category,source,forbidden_prompt" to "category,source,question"
    data_path = "strongreject/strongreject_dataset/strongreject_dataset.csv"
    df = pd.read_csv(data_path)
    df = df.rename(columns={"forbidden_prompt": "question"})
    print(f"Total instructions from StrongReject: {len(df)}, example instruction: {df['question'].iloc[0]}")
    df.to_csv("eval/safety/strongreject.csv", index=False, encoding="utf-8")
    
def process_wildteaming():
    file_path = 'wildjailbreak/eval/eval.tsv'
    content = pd.read_csv(file_path, sep="\t", header=0)
    content_malicious = content[content["label"] == 1]
    content_malicious_sampled = content_malicious.sample(n=500, random_state=42)
    # adversarial,label,data_type
    # question,label,source
    content_malicious_sampled = content_malicious_sampled.rename(columns={"adversarial": "question", "data_type": "source"})
    print(f"Sampled {len(content_malicious_sampled)} malicious instructions from Wildteaming, example instruction: {content_malicious_sampled['question'].iloc[0]}")
    content_malicious_sampled.to_csv("eval/safety/wildteaming.csv", index=False)
    
def process_eval_safety_data():
    os.makedirs("eval/safety", exist_ok=True)
    process_rrb()
    process_strongreject()
    process_wildteaming()
    
def process_eval_utility_data():
    os.makedirs("eval/utility", exist_ok=True)
    data_num = 500
    for dataset_name, split in [('bamboogle', 'test'), ('hotpotqa', 'dev'), ('triviaqa', 'test')]:
        test_path = f'./FlashRAG_datasets/{dataset_name}/{split}.jsonl'
        output_path = f'./eval/utility/{dataset_name}.json'

        data_list = []
        with open(test_path, 'r') as file:
            for id, line in enumerate(tqdm(file.readlines())):
                line = json.loads(line)
                data_list.append({
                    'id': id, 
                    'Question': line['question'],
                    'answer': line["golden_answers"],
                })
                if len(data_list) >= data_num:
                    break
        print(f"Processed {len(data_list)} samples from {dataset_name}, example question: {data_list[0]['Question']}")
        # Write the updated data to JSON
        with open(output_path, mode='w', encoding='utf-8') as json_file:
            json.dump(data_list, json_file, indent=4, ensure_ascii=False)


def get_data(id_list, content, split='train'):
    data = []
    for id_s in tqdm(id_list):
        _id, data_type, _, _ = id_s.split("_")
        row = content.iloc[int(_id)]
        instruction = row[data_type]
        
        data.append({
                "data_source": 'safe',
                "prompt": [{
                    "role": "user",
                    "content": get_search_agent_instruction(3, instruction),
                }],
                "ability": "safe",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "",
                    "question": instruction,
                },
                "extra_info": {
                    'split': split,
                    'index': _id,
                    'source_info': id_s
                }
            })
    return data
    
def process_finetune_safety_data():
    file_path = "wildjailbreak/train/train.tsv"
    content = pd.read_csv(file_path, sep="\t", header=0)
    
    train_ids_file = "finetune/safety/train_id.csv"
    test_ids_file = "finetune/safety/test_id.csv"
    train_ids = pd.read_csv(train_ids_file)['id'].tolist()
    test_ids = pd.read_csv(test_ids_file)['id'].tolist()
    
    
    train_data = get_data(train_ids, content)
    test_data = get_data(test_ids, content, split='test')
    
    # save to finetune/safety/train.parquet and finetune/safety/test.parquet
    pd.DataFrame(train_data).to_parquet("finetune/safety/train.parquet", index=False)
    pd.DataFrame(test_data).to_parquet("finetune/safety/test.parquet", index=False)
    print(f"Processed finetune safety data: {len(train_data)} training samples, {len(test_data)} testing samples.")

def transform(df, split='train'):
    transformed_data = []
    for row in df.iterrows():
        idx = row[0]
        row = row[1]
        instruction = row['question']

        transformed_data.append({
            "data_source": row['data_source'],
            "prompt": [{
                "role": "user",
                "content": get_search_agent_instruction(3, instruction),
            }],
            "ability": row['ability'],
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                "target": row['golden_answers'],
            }
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        })
    return transformed_data      
    

def process_finetune_utility_data():
    os.makedirs("finetune/utility", exist_ok=True)
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
    data_source = 'nq'
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    def make_map_fn(split):
        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": get_search_agent_instruction(3, example['question']),
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": example['golden_answers'],
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    train_dataset.to_parquet("finetune/utility/train.parquet")
    test_dataset.to_parquet("finetune/utility/test.parquet")

    print(f"Processed finetune utility data: {len(train_dataset)} training samples, {len(test_dataset)} testing samples.")
    # train_dataset = dataset['train']
    # test_dataset = dataset['test']
    
    # train_data = pd.read_parquet(train_data_path)
    # test_data = pd.read_parquet(test_data_path)
    
    # train_data = transform(train_data)
    # test_data = transform(test_data, split='test')
    # pd.DataFrame(train_data).to_parquet("finetune/utility/train.parquet", index=False)
    # pd.DataFrame(test_data).to_parquet("finetune/utility/test.parquet", index=False)
    # print(f"Processed finetune utility data: {len(train_data)} training samples, {len(test_data)} testing samples.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True, help="Type of data to process", choices=["eval_safety", "eval_utility", "finetune_safety", "finetune_utility"])
    args = parser.parse_args()
    if args.data_type == "eval_safety":
        process_eval_safety_data()
    elif args.data_type == "eval_utility":
        process_eval_utility_data()
    elif args.data_type == "finetune_safety":
        process_finetune_safety_data()
    elif args.data_type == "finetune_utility":
        process_finetune_utility_data()
    else:
        raise ValueError("Invalid data type specified.")