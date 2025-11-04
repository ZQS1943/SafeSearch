import re
import json
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict


def extract_answer(output):
    extracted_text = ''

    if 'assistantfinal' in output:
        output = output.split('assistantfinal')[-1]
    pattern = r'\\boxed\{(.*)\}'
    matches = re.findall(pattern, output)
    if matches:
        extracted_text = matches[-1]  # Take the last match

        inner_pattern = r'\\text\{(.*)\}'
        inner_matches = re.findall(inner_pattern, extracted_text)
        if inner_matches:
            extracted_text = inner_matches[-1]  # Take the last match
        extracted_text = extracted_text.strip("()")
    else:
        pattern = r'<answer>(.*)</answer>'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
                
    return extracted_text


def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_predictions(output, labeled_answer):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    pred_answer = extract_answer(output)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    normalized_pred_answer = normalize_answer_qa(pred_answer)
    for answer in labeled_answer:
        normalized_ground_truth = normalize_answer_qa(answer)
        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)

        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        for k in ["em", "acc", "f1"]:
            final_metric[k] = max(eval(k), final_metric[k])

    return final_metric, pred_answer



def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False):
    # Existing evaluation for other datasets
    avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
    num_valid_answer = 0

    # If the dataset is GPQA, track metrics per domain
    domain_metrics = {}

    for item, input_prompt, result in zip(filtered_data, input_list, output_list):
        if type(result) == str:
            item['Output'] = result
        else:
            item['Output'] = result.outputs[0].text
        
        labeled_answer = item["answer"]

        metric, pred_answer = evaluate_predictions(output=item['Output'], labeled_answer=labeled_answer)
        item['Metrics'] = metric
        item['Question'] = input_prompt

        # Determine the validity of the predicted answer
        my_method_valid = (pred_answer != '')
        avg_em.append(metric['em'])
        avg_acc.append(metric['acc'])
        avg_f1.append(metric['f1'])
        avg_math.append(metric['math_equal'])

        if my_method_valid:
            num_valid_answer += 1

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

    # Compute overall metrics
    overall_results = {
        'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
        'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
        'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
        'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
        'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
    }

    # 保存总体和分domain的指标
    final_metrics = {'overall': overall_results}

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')

    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)


