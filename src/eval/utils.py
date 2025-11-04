# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
import os
import re
import time
import json
from typing import Optional, Dict, Any

import pandas as pd
from src.utils.prompts import (
    COMBINED_PROMPT,
)

def parse_qscore(text: str) -> int:
    """
    Parse <qscore>±1</qscore> from model output; return 0 on failure.
    """
    if not text:
        return 0
    m = re.search(r"<qscore>\s*(-?1)\s*</qscore>", text)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def parse_dscore(text: str) -> int:
    """
    Parse <dscore>±1</dscore> from model output; return 0 on failure.
    """
    if not text:
        return 0
    m = re.search(r"<dscore>\s*(-?1)\s*</dscore>", text)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """
    Extract the last occurrence of content between start_tag and end_tag (non-greedy).
    """
    if not text:
        return None
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def make_jsonable(obj: Any):
    """
    Recursively convert pandas, sets, numpy scalars, etc. to JSON-serializable forms.
    """
    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, set):
        return list(obj)

    # numpy scalars
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except ModuleNotFoundError:
        pass

    # containers
    if isinstance(obj, list):
        return [make_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [make_jsonable(x) for x in obj]  # JSON has no tuple
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}

    return obj  # assume primitive


def read_cache(args) -> Dict[str, Any]:
    """
    Read search cache from args.search_cache_path.
    """
    path = args.search_cache_path
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CACHE] Error reading {path}: {e}. Start with empty cache.")
    return {}


def save_caches(args, search_cache: Dict[str, Any]) -> None:
    """
    Save search cache to args.search_cache_path.
    """
    path = args.search_cache_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(search_cache, f, ensure_ascii=False, indent=2)


def build_prompt(q: str, a: str) -> str:
    """
    Build a judging prompt that expects a single tag: <score>k</score>.
    """
    convo = f"User: {q}\n\nAssistant: {a}"
    if len(convo) > 10000:
        print(f"[build_prompt] conversation length = {len(convo)}")
    return COMBINED_PROMPT.format(conversation=convo)
