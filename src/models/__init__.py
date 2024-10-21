import importlib.util
from typing import List
import sys

__all__: List[str] = []

def is_package_installed(package_name: str) -> bool:
    spec = importlib.util.find_spec(package_name)
    return spec is not None

# Transformers
if is_package_installed("transformers"):
    from .transformers_assistant import TransformersAIAssistant
    __all__.append("TransformersAIAssistant")

# vLLM
if is_package_installed("vllm"):
    from .vllm_assistant import VLLMAIAssistant
    __all__.append("VLLMAIAssistant")

# RWKV
if is_package_installed("rwkv"):
    from .rwkv_assistant import RWKVAIAssistant
    __all__.append("RWKVAIAssistant")

# llama.cpp
if is_package_installed("llama_cpp"):
    from .llamacpp_assistant import LlamaCppAIAssistant
    __all__.append("LlamaCppAIAssistant")

# MLX
if is_package_installed("mlx"):
    from .mlx_assistant import MLXAIAssistant
    __all__.append("MLXAIAssistant")

# Local-Gemma
if is_package_installed("local_gemma"):
    from .localgemma_assistant import LocalGemmaAIAssistant
    __all__.append("GemmaAIAssistant")

# Print available assistants for debugging
print(f"Available AI Assistants: {', '.join(__all__)}")

