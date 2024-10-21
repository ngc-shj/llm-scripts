import io
import os
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""

setup(
    name="llm-scripts",
    version="0.1.0",
    author="NOGUCHI, Shoji",
    author_email="github.com@jpng.jp",
    license="Apache License 2.0",
    description="A versatile Python project designed to interact with "
                "various Large Language Models (LLMs) using different backends.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "huggingface_hub",
    ],
    extras_require={
        "transformers": [
            "accelerate",
            "bitsandbytes",
        ],
        "vllm": [
            "vllm",
        ],
        "rwkv": [
            "rwkv",
        ],
        "llama-cpp": [
            "llama-cpp-python",
        ],
        "mlx": [
            "mlx-lm",
        ],
        "local-gemma": [
            "accelerate",
            "local-gemma[cuda]",
        ],
        "all": [
            "vllm",
            "rwkv",
            "llama-cpp-python",
            "mlx-lm",
            "accelerate",
            "local-gemma[cuda]",
        ],
    },
)
