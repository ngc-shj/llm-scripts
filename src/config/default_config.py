DEFAULT_CONFIG = {
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "system_prompt": "あなたは誠実で優秀な日本人のアシスタントです。",
    "model_specific": {
        "transformers": {
            "device_map": "auto",
            "torch_dtype": "bfloat16",
        },
        "vllm": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
        },
        "localgemma": {
            "preset": "auto",
        },
        "rwkv": {
            "strategy": "cuda fp16",
            "chunk_len": "256",
        },
        "llamacpp": {
        },
        "mlx": {
            "repetition_context_size": 20,
        },
    }
}

def get_config(model_type: str):
    """Get the configuration for a specific model type."""
    config = DEFAULT_CONFIG.copy()
    config.update(DEFAULT_CONFIG["model_specific"].get(model_type, {}))
    return config

