# llm-scripts

## Overview

llm-scripts is a versatile Python project designed to interact with various Large Language Models (LLMs) using different backends. It provides a unified interface for running inference on models from popular frameworks such as Transformers, vLLM, RWKV, llama.cpp, MLX, and Local-Gemma.

## Features

- Support for multiple LLM backends:
  - Transformers
  - vLLM
  - RWKV
  - llama.cpp
  - MLX
  - Local-Gemma
- Unified command-line interface for all supported models
- Configurable system prompts and chat modes
- Extensible architecture for easy addition of new model backends
- Performance statistics output (tokens per second, total time)

## Project Structure

```
llm-scripts/
│
├── scripts/
│   ├── run_transformers.py
│   ├── run_vllm.py
│   ├── run_rwkv.py
│   ├── run_llamacpp.py
│   ├── run_mlx.py
│   └── run_localgemma.py
│
├── src/
│   ├── base/
│   │   ├── __init__.py
│   │   └── base_ai_assistant.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── default_config.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformers_assistant.py
│   │   ├── vllm_assistant.py
│   │   ├── rwkv_assistant.py
│   │   ├── llamacpp_assistant.py
│   │   ├── mlx_assistant.py
│   │   └── localgemma_assistant.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── argument_parser.py
│   │   ├── torch_utils.py
│   │   └── text_processing.py
│   │
│   └── __init__.py
│
├── setup.py
└── README.md
```

## Installation

llm-scripts uses a `setup.py` file to manage dependencies and provide flexible installation options. This allows you to install only the backends you need.

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-scripts.git
   cd llm-scripts
   ```

2. Install the base package (includes Transformers support):
   ```
   pip install .
   ```

### Optional Dependencies

You can install additional backends by specifying them as extras:

- For vLLM support:
  ```
  pip install ."[vllm]"
  ```

- For RWKV support:
  ```
  pip install ."[rwkv]"
  ```

- For llama.cpp support:
  ```
  pip install ."[llama-cpp]"
  ```

- For MLX support:
  ```
  pip install ."[mlx]"
  ```

- For Local-Gemma support:
  ```
  pip install ."[local-gemma]"
  ```

- To install all backends:
  ```
  pip install ."[all]"
  ```

You can also combine multiple backends in a single installation command:
```
pip install ."[vllm,rwkv]"
```

Note: Some backends may require additional system-level dependencies or CUDA installations. Please refer to their respective documentation for detailed instructions.

## Usage

To run inference with a specific model, use the corresponding script in the `scripts/` directory. For example:

```
python scripts/run_transformers.py --model-path /path/to/your/model --max-tokens 256
```

### Common Arguments

- `--model-path`: Path to the model (required)
- `--tokenizer-path`: Path to the tokenizer (if different from model path)
- `--no-use-system-prompt`: Disable the default system prompt
- `--max-tokens`: Maximum number of tokens to generate (default: 256)
- `--no-chat`: Disable chat mode
- `--temperature`: Temperature for sampling (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 40)
- `--repetition-penalty`: Repetition penalty (default: 1.1)

### Backend-Specific Arguments

#### Transformers
- `--device-map`: Device map for model parallelism (default: "auto")

#### vLLM
- `--tensor-parallel-size`: Tensor parallel size (default: 1)
- `--gpu-memory-utilization`: GPU memory utilization (default: 0.7)

#### RWKV
- `--model-file`: RWKV model filename
- `--strategy`: Strategy for RWKV model (default: "cuda fp16")
- `--chunk-len`: Chunk length for RWKV generation (default: 256)

#### llama.cpp
- `--ggml-model-path`: Path to the GGML model directory
- `--ggml-model-file`: GGML model file name
- `--n-ctx`: Context size (default: 2048)
- `--n-gpu-layers`: Number of GPU layers (default: 0)
- `--n-threads`: Number of threads (default: 2)

#### Local-Gemma
- `--preset`: Memory optimization preset (default: "auto")

## Extending the Project

To add support for a new LLM backend:

1. Create a new file in `src/models/` (e.g., `new_backend_assistant.py`)
2. Implement a new class that inherits from `BaseAIAssistant`
3. Override the necessary methods (`_load_model_and_tokenizer`, `_generate_response`)
4. Update `src/models/__init__.py` to include the new assistant class
5. Create a new script in the `scripts/` directory to run the new backend
6. Add the new backend's dependencies to `setup.py` in the `extras_require` dictionary

## Configuration

Default configurations for each backend can be found in `src/config/default_config.py`. You can modify these settings or override them using command-line arguments.

## Performance Statistics

After each inference, the script will output performance statistics, including:
- Number of input tokens
- Number of output tokens
- Tokens per second (TPS)
- Total execution time

## Contributing

Contributions to llm-scripts are welcome! Please feel free to submit pull requests, create issues, or suggest new features.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project utilizes various open-source libraries and models. We would like to thank the developers and contributors of Transformers, vLLM, RWKV, llama.cpp, MLX, and Local-Gemma for their excellent work.

