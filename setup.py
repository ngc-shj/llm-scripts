from setuptools import setup, find_packages

setup(
    name="llm-scripts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "huggingface_hub",
    ],
    extras_require={
        "vllm": ["vllm"],
        "rwkv": ["rwkv"],
        "llama-cpp": ["llama-cpp-python"],
        "mlx": ["mlx-lm"],
        "local-gemma": ["local-gemma[cuda]"],
        "all": [
            "vllm",
            "rwkv",
            "llama-cpp-python",
            "mlx-lm",
            "local-gemma[cuda]",
        ],
    },
)
