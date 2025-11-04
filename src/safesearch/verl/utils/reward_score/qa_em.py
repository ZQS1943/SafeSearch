import re
import string
import random
from src.utils.prompts import COMBINED_PROMPT, QUERY_PROMPT, END_SEARCH_RESULT, BEGIN_SEARCH_RESULT

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str: str):
    """Extract the equation from the solution string.
    
    Returns:
        (answer, has_format_issue)
        - answer: 提取出的内容 (str or None)
        - has_format_issue: 是否存在格式问题 (bool)，例如 0 个或多于 1 个匹配
    """
    # 切掉前面的标记
    solution_str = solution_str.split("<|im_start|>assistant")[-1]
    if END_SEARCH_RESULT in solution_str:
        solution_str = solution_str.split(END_SEARCH_RESULT)[-1]

    # 只找 \boxed{...}
    pattern = r'\\boxed\{(.*?)\}'
    matches = re.findall(pattern, solution_str, flags=re.DOTALL)  # 允许跨行 + 非贪婪

    if not matches:
        print(f"[WARN] No matches in: {solution_str}")
        return None, True   # 没有匹配 → 格式问题
    if len(matches) > 1:
        print(f"[WARN] Multiple matches in: {matches}")
        return None, True   # 多个匹配 → 格式问题

    # 唯一匹配
    extracted_text = matches[0]

    # 处理 \text{...} 包裹的情况
    inner_pattern = r'\\text\{(.*?)\}'
    inner_matches = re.findall(inner_pattern, extracted_text, flags=re.DOTALL)
    if inner_matches:
        extracted_text = inner_matches[-1]

    extracted_text = extracted_text.strip("()")
    return extracted_text, False
def compute_score_em(solution_str, reward_info,
                     method='strict',
                     format_score=0., score=1., format_penalty=-0.1):
    """
    返回更丰富的元信息，兼容旧接口。
    New return:
        (score, answer, has_format_issue, matched)
    Old return (backwards compatible):
        float score
    """
    ground_truth = reward_info.get('ground_truth') if isinstance(reward_info, dict) else None
    answer, has_format_issue = extract_solution(solution_str=solution_str)

    base_score = format_penalty if has_format_issue else 0.0

    matched = False
    if ground_truth is not None and answer is not None:
        try:
            matched = em_check(answer, ground_truth.get('target', ''))
        except Exception:
            matched = False

    if answer is None:
        # format_penalty as before
        scalar = format_penalty
    elif matched:
        scalar = score + base_score
    else:
        scalar = format_score + base_score

    # 返回 tuple（向后兼容，调用端可以只使用第一个元素或直接接受 float）
    return float(scalar), answer, bool(has_format_issue), bool(matched)


# def compute_score_em(solution_str, reward_info,
#                      method='strict',
#                      format_score=0., score=1., format_penalty=-0.1):
    
#     ground_truth = reward_info['ground_truth']
#     answer, has_format_issue = extract_solution(solution_str=solution_str)

#     base_score = format_penalty if has_format_issue else 0.0

#     if answer is None:
#         return format_penalty

#     if em_check(answer, ground_truth['target']):
#         return score + base_score
#     else:
#         return format_score + base_score



def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

