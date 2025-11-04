# Export the distributed parallel state from vLLM 0.10.x
from vllm.distributed import parallel_state  # noqa: F401

__all__ = ["parallel_state"]
