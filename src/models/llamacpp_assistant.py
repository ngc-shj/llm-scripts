import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, llama_chat_format
from src.base import BaseAIAssistant
from typing import List, Dict, Union, Tuple
import time

class LlamaCppAIAssistant(BaseAIAssistant):
    def __init__(self, args, config):
        super().__init__(args, config)

    def _load_model_and_tokenizer(self):
        local_model_path = os.path.join(self.args.ggml_model_path, self.args.ggml_model_file)
        if os.path.isfile(local_model_path):
            ggml_model_path = local_model_path
        else:
            ggml_model_path = hf_hub_download(
                self.args.ggml_model_path,
                filename=self.args.ggml_model_file
            )

        self.chat_formatter = llama_chat_format.hf_autotokenizer_to_chat_formatter(self.args.model_path)
        chat_handler = llama_chat_format.hf_autotokenizer_to_chat_completion_handler(self.args.model_path)

        model = Llama(
            model_path=ggml_model_path,
            chat_handler=chat_handler,
            n_ctx=self.config.get("n_ctx", 2048),
            n_parts=self.config.get("n_parts", -1),
            n_gpu_layers=self.args.n_gpu_layers,
            seed=self.config.get("seed", -1),
            f16_kv=self.config.get("f16_kv", True),
            logits_all=self.config.get("logits_all", False),
            vocab_only=self.config.get("vocab_only", False),
            use_mlock=self.config.get("use_mlock", False),
            n_threads=self.config.get("n_threads", None),
            n_batch=self.config.get("n_batch", 512),
            last_n_tokens_size=self.config.get("last_n_tokens_size", 64),
            lora_base=self.config.get("lora_base", None),
            lora_path=self.config.get("lora_path", None),
            verbose=self.config.get("verbose", True)
        )
        return model, model

    def _generate_response(self, messages: Union[List[Dict[str, str]], str]) -> Tuple[str, int, int, float, float]:
        if self.is_chat:
            prompt = self.chat_formatter(messages=messages)
        else:
            prompt = messages

        start_time = time.time()

        output = self.model(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
            top_k=self.config.get("top_k", 40),
            repeat_penalty=self.config.get("repeat_penalty", 1.1),
            stream=True
        )

        generated_text = ""
        for chunk in output:
            chunk_text = chunk["choices"][0]["text"]
            print(chunk_text, end="", flush=True)
            generated_text += chunk_text

        end_time = time.time()

        input_tokens = len(self.tokenizer.tokenize(prompt.encode('utf-8')))
        output_tokens = len(self.tokenizer.tokenize(generated_text.encode('utf-8')))

        return generated_text, input_tokens, output_tokens, start_time, end_time

