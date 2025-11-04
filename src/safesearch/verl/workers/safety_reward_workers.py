# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
import logging
import os
import torch
from src.safesearch.verl import DataProto
from src.safesearch.verl.single_controller.base import Worker
from src.safesearch.verl.single_controller.base.decorator import register, Dispatch
from src.safesearch.verl.utils.reward_score.qa_em import extract_solution
import re
import json
import time
import requests 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
load_dotenv() 

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

from src.utils.prompts import COMBINED_PROMPT, QUERY_PROMPT, END_SEARCH_RESULT, BEGIN_SEARCH_RESULT, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY
from src.utils.llm import _OPENAIClient

def extract_final_response(solution_str):
    if END_SEARCH_RESULT in solution_str:
        return solution_str.split(END_SEARCH_RESULT)[-1].strip()
    return solution_str.split('<|im_start|>assistant')[-1].strip()

def build_prompt_final_eval(q, a):
    convo = f"User: {q}\n\nAssistant: {a}"
    if len(convo) > 10000:
        print(a)
        print(q)
        print(f"Building prompt :{len(convo)} characters")
    return COMBINED_PROMPT.format(conversation=convo)


class SafetyRewardWorker(Worker):
    """
    Safety/Helpfulness reward model that uses a remote vLLM server
    (OpenAI-compatible /v1/chat/completions) instead of loading weights locally.
    - No FSDP/Megatron
    - GPU allocation handled by upstream actor pool for the *policy*; this worker does HTTP calls only.
    - Returns token-level rm_scores (safe=1.0 / unsafe=0.0 shaped) as before.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.client = _OPENAIClient()
        
        self.w_helpfulness = config.get("w_helpfulness", True) 
        self.w_query_score = config.get("w_query_score", True)
        self.safe_reward   = config.get("safe_reward", 1.0)
        self.unsafe_reward = config.get("unsafe_reward", 0.0)
        self.max_new_tokens  = config.get("max_new_tokens", 512)

        self.lambda_s = config.get("lambda_s", 0.25)
        self.lambda_q = config.get("lambda_q", 0.15)  
        self.gamma = config.get("gamma", 0.95)       
        self.q_pos = config.get("q_pos", 1.0) 
        self.q_neg = config.get("q_neg", -3.0) 
        
        self.max_queries = config.get("max_queries", 3)
        self.pre_query_min_tokens = config.get("pre_query_min_tokens", 24)


        print(f"[DEBUG] Reward params: safe {self.safe_reward}, unsafe {self.unsafe_reward}, lambda_s {self.lambda_s}, lambda_q {self.lambda_q}, gamma {self.gamma}, q_pos {self.q_pos}, q_neg {self.q_neg}")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, tokenizer=None):
        """Tokenizer is provided by upstream (policy). We need it for decode/offsets."""
        self.tokenizer = tokenizer
        return

    def _find_query_char_spans_strict(self, text: str):
        """
        Strictly extract query spans:
        - Only accept exact pairs: <|begin_search_query|> ... <|end_search_query|>
        - Non-greedy across newlines (DOTALL)
        - Reject if the inner content contains another '<|begin_search_query|>' (nested)
        - Reject if inner contains any '<|' tag fragments (e.g., other tags are embedded)
        Returns:
        spans: list of (tag_start, left, right) char indices for the cleaned content
        malformed: int, number of malformed occurrences encountered
        """
        spans = []
        malformed = 0

        pat = re.compile(r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>", re.DOTALL)
        for m in pat.finditer(text):
            tag_start = m.start(0)
            inner = m.group(1) 
            s = m.start(1); e = m.end(1)

            left  = s + (len(inner) - len(inner.lstrip()))
            right = e - (len(inner) - len(inner.rstrip()))
            frag = text[left:right]

            if BEGIN_SEARCH_QUERY in frag:
                malformed += 1
                continue
            if "<|" in frag:  
                malformed += 1
                continue

            if right > left:
                spans.append((tag_start, left, right))
            else:
                malformed += 1

        begin_only = len(re.findall(r"<\|begin_search_query\|>", text)) - len(spans)
        if begin_only > 0:
            malformed += begin_only

        return spans, malformed

    
    
    def _find_result_block_spans(self, text: str):
        """
        Return list of (start_char, end_char) spans for full <result> blocks,
        i.e., from <|begin_search_result|> to <|end_search_result|>.
        """
        spans = []
        for m in re.finditer(r"<\|begin_search_result\|>.*?<\|end_search_result\|>", text, flags=re.DOTALL):
            spans.append((m.start(), m.end()))
        return spans

    def _count_tokens_in_substring(self, text: str, char_start: int, char_end: int) -> int:
        """
        Count tokenizer tokens in text[char_start:char_end] (exclusive). 
        We re-tokenize the substring to avoid boundary drift.
        """
        if char_end <= char_start:
            return 0
        sub = text[char_start:char_end]
        enc = self.tokenizer(sub, add_special_tokens=False)
        return len(enc["input_ids"])

    def _gather_query_infos_for_sample(self, full_text: str):
        spans, malformed = self._find_query_char_spans_strict(full_text) 
        if not spans:
            return [], malformed

        result_ends = [re for (_, re) in self._find_result_block_spans(full_text)]
        result_ends.sort()

        infos = []
        for t, (tag_start, cs, ce) in enumerate(spans):
            qtext = full_text[cs:ce].strip()
            if not qtext:
                malformed += 1
                continue

            token_idx = self._char_end_to_token_idx(full_text, ce)
            turn_start = 0
            for re_pos in result_ends:
                if re_pos <= cs:
                    turn_start = re_pos
                else:
                    break
            pre_tok = self._count_tokens_in_substring(full_text, turn_start, tag_start)

            infos.append({
                "cs": cs, "ce": ce,
                "token_idx": token_idx,
                "t": t,
                "q": qtext,
                "pre_tokens": pre_tok,
            })
        return infos, malformed



    def _char_end_to_token_idx(self, full_text: str, end_char_exclusive: int) -> int:
        enc = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        offsets = enc["offset_mapping"]
        target_char = max(0, end_char_exclusive - 1)
        last_token_idx = 0
        for i, (s, e) in enumerate(offsets):
            if s <= target_char < e:
                last_token_idx = i
                break
            if e <= target_char:
                last_token_idx = i
        return last_token_idx

    def _score_queries_batch(self, queries):
        """
        For each search query string, ask the remote model to emit <qscore>1</qscore> or <qscore>-1</qscore>.
        Returns list of floats in {-1.0, 0.0, +1.0}.
        """
        if not queries:
            return []
        scores = []
        cur_time = time.time()
        query_prompts = [QUERY_PROMPT + "\n\n" + q.strip() for q in queries]
        outs = self.client.chat_multiple(query_prompts, max_tokens=self.max_new_tokens)
        for q, text in zip(queries, outs):
            print(f"[DEBUG] Query: {q}")
            print(f"[DEBUG] Query response: {text}")
            val = 0
            if text is None:
                text = ""
            if "<qscore>" in text and "</qscore>" in text:
                try:
                    val = int(text.split("<qscore>")[-1].split("</qscore>")[0].strip())
                    val = max(-1, min(1, val))
                except Exception:
                    val = 0
            print(f"[DEBUG] Query score: {val}")
            if val == 1:
                scores.append(float(self.q_pos))
            elif val == -1:
                scores.append(float(self.q_neg))
            else:
                scores.append(0.0)
        minutes = max(1e-9, (time.time() - cur_time) / 60)
        print(f"[DEBUG] Processed {len(queries)} queries in {minutes:.2f} min ({len(queries) / minutes:.2f} q/min)")
        return scores

    def _score_final_outputs_batch(self, items):
        """
        Returns:
            out_scores: List[float]  # raw 1-4 (or unsafe_reward for unsafe) or 0.0 if no tag
            empty_flags: List[int]   # 1 if no <score>/<safety> tag was parsed, else 0
        """
        if not items:
            return [], []
        out_scores, empty_flags = [], []
        cur_time = time.time()
        user_msgs = [item["text"][0]["content"] for item in items]
        outputs = self.client.chat_multiple(user_msgs, max_tokens=self.max_new_tokens)
        for item, text in zip(items, outputs):
            score, empty = 0.0, 0
            if text is None:
                text = ""
            if "<safety>unsafe</safety>" in text:
                score = self.unsafe_reward
            elif "<score>" in text and "</score>" in text:
                if self.w_helpfulness:
                    try:
                        score = int(text.split("<score>")[-1].split("</score>")[0].strip())
                    except Exception:
                        score = 0.0
                else:
                    score = self.safe_reward
            else:
                score, empty = 0.0, 1
            out_scores.append(float(score))
            empty_flags.append(int(empty))
        return out_scores, empty_flags

    
    

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to('cpu')
        bs, resp_len = data.batch['responses'].shape
        attn     = data.batch['attention_mask']
        prompts  = data.batch['prompts']
        responses= data.batch['responses']
        source   = data.non_tensor_batch['data_source']
        q_texts  = data.non_tensor_batch['question']

        resp_texts_full, final_texts = [], []
        for i in range(bs):
            r_ids = responses[i]
            full  = self.tokenizer.decode(r_ids, skip_special_tokens=False)
            no_sp = self.tokenizer.decode(r_ids, skip_special_tokens=True)
            resp_texts_full.append(full)
            final_texts.append(extract_final_response(no_sp))

        final_items, final_map = [], []
        format_ok = {}  
        for i, (q, fin, s) in enumerate(zip(q_texts, final_texts, source)):
            if s != 'safe':
                continue
            _, has_fmt_issue = extract_solution(solution_str=fin)
            if (BEGIN_SEARCH_QUERY in fin) or (BEGIN_SEARCH_RESULT in fin) or has_fmt_issue:
                format_ok[i] = 0
            else:
                format_ok[i] = 1
            final_items.append({"idx": i, "text": [{"role": "user",
                                                    "content": build_prompt_final_eval(q, fin)}]})
            final_map.append(i)

        judge_latency_s = 0.0
        if final_items:
            t0 = time.perf_counter()
            final_scores_raw, judge_empty_flags = self._score_final_outputs_batch(final_items)
            judge_latency_s = time.perf_counter() - t0
        else:
            final_scores_raw, judge_empty_flags = [], []

        safe_helpful_idx = {idx for idx, s in zip(final_map, final_scores_raw) if s >= 2}
        final_scores     = {idx: self.lambda_s * s for idx, s in zip(final_map, final_scores_raw)}

        per_query_by_sample = {}
        q_malformed_total = 0
        if self.w_query_score:
            for i, (full, s) in enumerate(zip(resp_texts_full, source)):
                if s != 'safe':
                    continue
                infos, malformed = self._gather_query_infos_for_sample(full)
                q_malformed_total += int(malformed or 0)
                
                if not infos:
                    continue
                for d in infos:
                    d["i"] = i
                per_query_by_sample[i] = infos

            to_score_infos = []
            for i, infos in per_query_by_sample.items():
                infos.sort(key=lambda x: x.get("t", 0))
                to_score_infos.extend(infos[:self.max_queries])

            all_queries = [x["q"] for x in to_score_infos]
            if all_queries:
                q_scores = self._score_queries_batch(all_queries) 
                assert len(q_scores) == len(to_score_infos)
            else:
                q_scores = []
            assert len(q_scores) == len(to_score_infos), "q_scores misaligned with to_score_infos"
        else:
            to_score_infos, q_scores = [], []

        token_level = torch.zeros((bs, resp_len), dtype=torch.float32)

        for i in range(bs):
            last_token_pos = max(0, int(attn[i, prompts.shape[1]:].sum().item()) - 1)
            if i in final_scores:
                token_level[i, last_token_pos] += min(final_scores[i],
                                                    final_scores[i] * float(format_ok.get(i, 1)))
            if not bool(format_ok.get(i, 1)):
                token_level[i, last_token_pos] -= float(getattr(self, "format_penalty", 0.02))

        q_ptr = 0
        early_blocked = 0
        q_pos_cnt = 0
        q_neg_cnt = 0
        if self.w_query_score:
            for info in to_score_infos:
                i         = info["i"]
                token_idx = info["token_idx"]
                t         = int(info.get("t", 0))
                pre_tok   = int(info.get("pre_tokens", 0))
                qscore    = float(q_scores[q_ptr]); q_ptr += 1

                if qscore > 0:
                    q_pos_cnt += 1
                    if i not in safe_helpful_idx:
                        continue
                    if pre_tok < self.pre_query_min_tokens:
                        token_level[i, token_idx] -= float(getattr(self, "early_query_penalty", 0.02))
                        early_blocked += 1
                        continue
                elif qscore < 0:
                    q_neg_cnt += 1

                shaped = self.lambda_q * (self.gamma ** t) * qscore
                token_level[i, token_idx] += shaped

        num_final        = len(final_scores_raw)
        final_hit_num    = int(sum(1 for s in final_scores_raw if s >= 2))
        unsafe_num       = int(sum(1 for s in final_scores_raw if s < 0))
        format_ok_num    = int(sum(format_ok.get(i, 1) for i in final_map))
        judge_empty_num  = int(sum(judge_empty_flags)) if judge_empty_flags else 0

        if self.w_query_score:
            pre_tokens_list      = [int(d.get("pre_tokens", 0)) for infos in per_query_by_sample.values() for d in infos]
            first_pre_tokens_list= [int(d.get("pre_tokens", 0)) for infos in per_query_by_sample.values() for d in infos if d.get("t", -1) == 0]
        else:
            pre_tokens_list, first_pre_tokens_list = [], []

        metrics = {
            # final / safety
            "rm/final_hit_num": final_hit_num,
            "rm/final_hit_den": int(num_final),
            "rm/unsafe_num":    unsafe_num,
            "rm/unsafe_den":    int(num_final),

            # format / judge
            "rm/format_ok_num":   format_ok_num,
            "rm/format_ok_den":   int(len(final_map)),
            "rm/judge_empty_num": judge_empty_num if num_final else 0,
            "rm/judge_empty_den": int(len(judge_empty_flags)) if num_final else 0,

            # queries
            "rm/q_pos_num":       int(q_pos_cnt),
            "rm/q_neg_num":       int(q_neg_cnt),
            "rm/q_to_score":      int(len(to_score_infos)) if self.w_query_score else 0,
            "rm/q_early_blocked": int(early_blocked),
            "rm/q_malformed_num": int(q_malformed_total),

            # pre-tokens
            "rm/q_pre_tokens_sum":       float(sum(pre_tokens_list)) if pre_tokens_list else 0.0,
            "rm/q_pre_tokens_cnt":       int(len(pre_tokens_list)),
            "rm/q_first_pre_tokens_sum": float(sum(first_pre_tokens_list)) if first_pre_tokens_list else 0.0,
            "rm/q_first_pre_tokens_cnt": int(len(first_pre_tokens_list)),

            # judge 
            "rm/judge_latency_sum": float(judge_latency_s),
            "rm/judge_latency_cnt": 1,
        }

        examples = []
        if getattr(self, "log_examples", True):
            final_raw_map = {idx: s for idx, s in zip(final_map, final_scores_raw)}
            pass_mask = set(i for i, s in final_raw_map.items() if s >= 2)

            per_sample_scored = {}
            if self.w_query_score:
                q_ptr2 = 0
                for ent in to_score_infos:
                    i = ent["i"]; e2 = dict(ent); e2["qscore"] = float(q_scores[q_ptr2]); q_ptr2 += 1
                    per_sample_scored.setdefault(i, []).append(e2)
                for i, infos in per_query_by_sample.items():
                    if len(infos) > self.max_queries:
                        for ent in infos[self.max_queries:]:
                            e2 = dict(ent); e2["qscore"] = None; e2["overlimit"] = True
                            per_sample_scored.setdefault(i, []).append(e2)

            def _clip(s):
                if s is None: return None
                s = str(s)
                lim = int(getattr(self, "max_field_chars", 4000))
                return s[:lim] + ("…[TRUNC]" if len(s) > lim else "")

            kept = 0
            cap  = int(getattr(self, "max_examples_per_step", 500))
            for i, (x, full, fin, s) in enumerate(zip(q_texts, resp_texts_full, final_texts, source)):
                if s != 'safe': 
                    continue
                if kept >= cap:
                    break
                ex = {
                    "sample_index": int(i),
                    "source": "safe",
                    "input": _clip(x),
                    "final_answer": _clip(fin),
                }
                if getattr(self, "include_full_output", True):
                    ex["output_full"] = _clip(full)

                raw = final_raw_map.get(i, None)
                ex["helpfulness_raw"] = float(raw) if raw is not None else None
                ex["final_scaled"]    = float(final_scores.get(i, 0.0)) if i in final_scores else None
                ex["final_pass_ge2"]  = bool(i in pass_mask)
                ex["format_ok"]       = bool(format_ok.get(i, 0))
                try:
                    ex["judge_empty"] = bool(judge_empty_flags[final_map.index(i)]) if judge_empty_flags else False
                except Exception:
                    ex["judge_empty"] = False

                qs = []
                if self.w_query_score:
                    ents = per_sample_scored.get(i, [])
                    for ent in sorted(ents, key=lambda z: z.get("t", 0)):
                        qtext   = ent.get("q", "")
                        pre_tok = int(ent.get("pre_tokens", 0))
                        t_ord   = int(ent.get("t", 0))
                        token_idx = int(ent.get("token_idx", -1))
                        qscore  = ent.get("qscore", None)
                        gated_final = (qscore is not None and qscore > 0 and i not in pass_mask)
                        gated_min   = (qscore is not None and qscore > 0 and pre_tok < self.pre_query_min_tokens)
                        qs.append({
                            "t": t_ord,
                            "text": _clip(qtext),
                            "pre_tokens": pre_tok,
                            "token_idx": token_idx,
                            "qscore": None if qscore is None else float(qscore),
                            "label": None if qscore is None else ("benign" if qscore > 0 else ("unsafe" if qscore < 0 else "neutral")),
                            "gated_final": bool(gated_final),
                            "gated_min_tokens": bool(gated_min),
                            "overlimit": bool(ent.get("overlimit", False)),
                        })
                ex["queries"] = qs
                ex["query_counts"] = {
                    "total": len(per_query_by_sample.get(i, [])) if self.w_query_score else 0,
                    "scored": len([1 for z in qs if z["qscore"] is not None]),
                    "overlimit": len([1 for z in qs if z.get("overlimit", False)]),
                    "pos": len([1 for z in qs if (z["qscore"] is not None and z["qscore"] > 0)]),
                    "neg": len([1 for z in qs if (z["qscore"] is not None and z["qscore"] < 0)]),
                    "early_blocked_pos": len([1 for z in qs if (z["qscore"] is not None and z["qscore"] > 0 and z["gated_min_tokens"])]),
                }
                examples.append(ex); kept += 1
        else:
            examples = []

        examples_aligned = [None] * bs
        for ex in examples:
            i = ex.get("sample_index", None)
            if i is not None and 0 <= i < bs:
                examples_aligned[i] = ex

        metrics_rank = [None] * bs
        metrics_rank[0] = metrics 

        out_tensors  = {'rm_scores': token_level}
        out_nontensor= {
            'rm_example':      np.array(examples_aligned, dtype=object),
            'rm_metrics_rank': np.array(metrics_rank,    dtype=object),
        }
        return DataProto.from_dict(out_tensors, out_nontensor)

    