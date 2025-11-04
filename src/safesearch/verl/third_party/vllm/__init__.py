from importlib.metadata import version, PackageNotFoundError

def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None

package_name = "vllm"
raw_version = get_version(package_name)

# normalize: keep only "major.minor.patch"
package_version = None
if raw_version:
    package_version = raw_version.split("+", 1)[0]  # e.g. "0.10.1+gptoss" -> "0.10.1"

if package_version == "0.3.1":
    vllm_version = "0.3.1"
    from .vllm_v_0_3_1.llm import LLM, LLMEngine
    from .vllm_v_0_3_1 import parallel_state
elif package_version == "0.4.2":
    vllm_version = "0.4.2"
    from .vllm_v_0_4_2.llm import LLM, LLMEngine
    from .vllm_v_0_4_2 import parallel_state
elif package_version == "0.5.4":
    vllm_version = "0.5.4"
    from .vllm_v_0_5_4.llm import LLM, LLMEngine
    from .vllm_v_0_5_4 import parallel_state
elif package_version in ("0.6.3",):
    vllm_version = "0.6.3"
    from .vllm_v_0_6_3.llm import LLM, LLMEngine
    from .vllm_v_0_6_3 import parallel_state
# elif package_version == "0.10.1":
#     vllm_version = "0.10.1"
#     from .vllm_v_0_10_1.llm import LLM, LLMEngine
#     from .vllm_v_0_10_1 import parallel_state
else:
    raise ValueError(
        f"vllm version {raw_version} not supported. "
        "Currently supported versions are 0.3.1, 0.4.2, 0.5.4, 0.6.3."
    )
