import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Assistant Configuration")
    
    # Common arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tokenizer-path", type=str, help="Path to the tokenizer (if different from model path)")
    parser.add_argument("--no-use-system-prompt", action='store_true', help="Do not use the default system prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--no-chat", action='store_true', help="Disable chat mode")
    
    # Transformers specific arguments
    parser.add_argument("--device-map", type=str, default="auto", help="Device map for model parallelism")
    
    # vLLM specific arguments
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (for vLLM)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="GPU memory utilization (for vLLM)")
    
    # RWKV specific arguments
    parser.add_argument("--model-file", type=str, help="RWKV model filename")
    parser.add_argument("--strategy", type=str, default="cuda fp16", help="Strategy for RWKV model")
    parser.add_argument("--chunk-len", type=int, default=256, help="Chunk length for RWKV generation")
    
    # llama.cpp specific arguments
    parser.add_argument("--ggml-model-path", type=str, help="Path to the GGML model directory")
    parser.add_argument("--ggml-model-file", type=str, help="GGML model file name")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size for llama.cpp")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="Number of GPU layers for llama.cpp")
    parser.add_argument("--n-threads", type=int, default=2, help="Number of threads for llama.cpp")
    
    # local gemma specific arguments
    parser.add_argument("--preset", type=str, default="auto", help="Memory Optimization for local-gemma")

    # Common generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")

    return parser.parse_args()

