# utils/llm.py
from __future__ import annotations

import os
from typing import Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from tqdm import tqdm


class _OPENAIClient:
    def __init__(
        self,
        model: str = "openai.gpt-oss-20b-1:0",
        base_url: Optional[str] = None,
        api_key_env: str = "AWS_BEARER_TOKEN_BEDROCK",
        max_workers: int = 16,
        request_timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Thin wrapper around an OpenAI-compatible Chat Completions endpoint.
        """
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {api_key_env}"
            )

        self.client = OpenAI(
            base_url=base_url or "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
            api_key=api_key,
            timeout=request_timeout,
        )
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries

    def _extract_text(self, completion) -> str:
        """
        Normalize various response shapes to a plain string.
        """
        try:
            msg = completion.choices[0].message
            content = msg.content
            if isinstance(content, str):
                return content
            # content can be a list of content parts; join text parts
            if isinstance(content, list):
                parts = []
                for c in content:
                    # each part may look like {"type": "text", "text": "..."}
                    if isinstance(c, dict) and "text" in c:
                        parts.append(c["text"])
                    elif isinstance(c, str):
                        parts.append(c)
                return "".join(parts)
            # fallback to string repr
            return str(content)
        except Exception:
            return ""

    def chat(self, prompt: str, max_tokens: int = 256, temperature: float = 0) -> str:
        """
        Single-turn chat. Retries on transient errors.
        """
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt < self.max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._extract_text(completion)
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_err = e
                attempt += 1
            except Exception as e:
                last_err = e
                break  # non-transient
        print(f"[JUDGE] chat failed after {self.max_retries} attempts: {last_err}")
        return ""

    def chat_multiple(self, prompts: Iterable[str], max_tokens: int = 256) -> List[str]:
        """
        Batch judge with threads. Returns one text per prompt (empty on failure).
        """
        prompts = list(prompts)
        results: List[Optional[str]] = [None] * len(prompts)

        def _run_one(idx: int, p: str) -> None:
            try:
                results[idx] = self.chat(p, max_tokens=max_tokens)
            except Exception as e:
                print(f"[JUDGE][Bedrock] error: {e}")
                results[idx] = ""

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_run_one, i, p) for i, p in enumerate(prompts)]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Judging prompts"):
                pass

        # cast Nones to empty strings
        return [r or "" for r in results]
