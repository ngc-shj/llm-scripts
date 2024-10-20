import torch
from mlx_lm import load, generate, stream_generate
from src.base import BaseAIAssistant
from typing import List, Dict, Union, Tuple
import time

class MLXAIAssistant(BaseAIAssistant):
    def __init__(self, args, config):
        super().__init__(args, config)

    def _load_model_and_tokenizer(self):
        return load(path_or_hf_repo=self.args.model_path)

    def _generate_response(self, messages: Union[List[Dict[str, str]], str]) -> Tuple[str, int, int, float, float]:
        generation_params = {
            "temp": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 0.9),
            "max_tokens": self.max_new_tokens,
            "repetition_penalty": self.config.get("repetition_penalty", 1.1),
            "repetition_context_size": self.config.get("repetition_context_size", 20),
        }

        if self.is_chat:
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = messages

        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            return_tensors="pt"
        )

        start_time = time.time()

        output = ""
        for t in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            **generation_params
        ):
            print(t, end="", flush=True)
            output += t
        print()

        end_time = time.time()

        output_ids = self.tokenizer.encode(
            output,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        input_tokens = len(input_ids[0])
        output_tokens = len(output_ids[0])

        return output, input_tokens, output_tokens, start_time, end_time

