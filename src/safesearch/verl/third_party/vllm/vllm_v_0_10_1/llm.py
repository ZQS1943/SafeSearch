from __future__ import annotations
import os
from typing import Any, Dict, Tuple
from vllm import LLM as _VLLM_LLM
from vllm import LLMEngine as _VLLM_Engine
import torch


class _R1WorkerExtension:
    """
    Runs INSIDE each vLLM worker process. Keep the method name stable.
    """
    def update_weights_from_tensors(self, named_tensors):
        # named_tensors: list[(name, torch.Tensor on CPU or CUDA)]
        weights = []
        for name, t in named_tensors:
            tt = t
            if not t.is_cuda:
                tt = t.to(device="cuda", non_blocking=True)
            weights.append((name, tt))
        # vLLM's internal hot path to replace parameters at runtime
        self.model_runner.model.load_weights(weights=weights)  # noqa
        torch.cuda.synchronize()

def _ensure_sleep_mode(kwargs: dict) -> dict:
    # Default to enabling sleep/wake APIs in vLLM 0.10.x
    kwargs.setdefault("enable_sleep_mode", True)
    return kwargs

_DROP_KEYS = {
    "model_hf_config",  # removed in vLLM 0.10.x
}
_RENAME_KEYS: Dict[str, str] = {}

def _sanitize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    for k in list(kwargs.keys()):
        if k in _DROP_KEYS:
            kwargs.pop(k, None)
    for old, new in list(_RENAME_KEYS.items()):
        if old in kwargs and new not in kwargs:
            kwargs[new] = kwargs.pop(old)
    return kwargs

def _extract_name_or_path(obj) -> str | None:
    # Try common Hugging Face fields
    for attr in ("_name_or_path", "name_or_path", "pretrained_model_name_or_path"):
        try:
            v = getattr(obj, attr, None)
            if isinstance(v, (str, os.PathLike)) and str(v):
                return str(v)
        except Exception:
            pass
    # Sometimes tokenizer has a .init_kwargs dict
    try:
        init = getattr(obj, "init_kwargs", None)
        if isinstance(init, dict):
            v = init.get("name_or_path") or init.get("pretrained_model_name_or_path")
            if isinstance(v, (str, os.PathLike)) and str(v):
                return str(v)
    except Exception:
        pass
    return None

def _split_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    # Convert first positional arg (HF/FSDP module) → model path for vLLM 0.10.x
    if len(args) == 0:
        return args, kwargs

    first = args[0]
    if isinstance(first, (str, os.PathLike)):
        return args, kwargs  # already a path

    model_id = None
    try:
        cfg = getattr(first, "config", None)
        if cfg is not None:
            model_id = _extract_name_or_path(cfg)
    except Exception:
        pass

    if model_id:
        args = args[1:]
        kwargs.setdefault("model", model_id)
        return args, kwargs

    if isinstance(kwargs.get("model"), (str, os.PathLike)):
        args = args[1:]
        return args, kwargs

    raise TypeError(
        "Search-R1 passed a model object to vLLM LLM, but vLLM>=0.10.x "
        "requires model=<str path or repo id>. Could not infer path."
    )

def _normalize_tokenizer(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure tokenizer kwarg is a string path/repo id or unset."""
    tok = kwargs.get("tokenizer", None)
    if tok is None:
        return kwargs

    if isinstance(tok, (str, os.PathLike)):
        return kwargs  # OK

    # Try to derive a path from tokenizer object
    tok_id = _extract_name_or_path(tok)
    if not tok_id:
        # Fall back to model path if present; otherwise drop it
        model_id = kwargs.get("model", None)
        if isinstance(model_id, (str, os.PathLike)):
            kwargs["tokenizer"] = str(model_id)
        else:
            kwargs.pop("tokenizer", None)  # let vLLM choose
    else:
        kwargs["tokenizer"] = tok_id
    return kwargs

class LLM(_VLLM_LLM):
    def __init__(self, *args, **kwargs):
        kwargs = _sanitize_kwargs(dict(kwargs))
        args, kwargs = _split_args(args, kwargs)
        kwargs = _normalize_tokenizer(kwargs)  # NEW
        kwargs = _ensure_sleep_mode(kwargs)
        kwargs.setdefault("worker_extension_cls", f"{__name__}._R1WorkerExtension")
        super().__init__(*args, **kwargs)
        
    @torch.no_grad()
    def sync_model_weights(self, params, *, to_dtype=None, shard_filter=None):
        """
        Minimal, naive sync: push CPU tensors through driver -> workers.
        `params` can be a dict[name->Tensor] or an iterator of (name, Tensor).
        """
        if isinstance(params, dict):
            named = list(params.items())
        else:
            # maybe a module.parameters() or named_parameters()
            try:
                # named_parameters()
                named = list(params)
            except TypeError:
                raise ValueError("Unsupported params type for sync_model_weights")

        # Materialize to CPU to avoid device mismatch / FSDP sharding surprises
        out = []
        for name, tensor in named:
            t = tensor.detach()
            if shard_filter and not shard_filter(name):
                continue
            if to_dtype is not None:
                t = t.to(dtype=to_dtype)
            out.append((name, t.cpu()))

        # Broadcast to all vLLM workers (Ray RPC fan-out). This is slow for 7B+.
        self.llm_engine.model_executor._run_workers(
            "update_weights_from_tensors", args=(out,)
        )

class LLMEngine(_VLLM_Engine):
    def __init__(self, *args, **kwargs):
        kwargs = _sanitize_kwargs(dict(kwargs))
        kwargs = _normalize_tokenizer(kwargs)  # NEW (defensive)
        kwargs = _ensure_sleep_mode(kwargs)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(cls, *args, **kwargs):
        kwargs = _sanitize_kwargs(dict(kwargs))
        kwargs = _normalize_tokenizer(kwargs)  # NEW (defensive)
        return _VLLM_Engine.from_engine_args(*args, **kwargs)
