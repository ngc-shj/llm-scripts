import os
from huggingface_hub import hf_hub_download
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from src.base import BaseAIAssistant
from typing import List, Dict, Union, Tuple
import time

class RWKVAIAssistant(BaseAIAssistant):
    def __init__(self, args, config):
        super().__init__(args, config)

    def _load_model_and_tokenizer(self):
        local_model_path = os.path.join(self.args.model_path, self.args.model_file)
        if os.path.isfile(local_model_path):
            model_path = local_model_path
        else:
            model_path = hf_hub_download(self.args.model_path, filename=self.args.model_file)
        model = RWKV(model=model_path, strategy=self.config.get("strategy", "cuda fp16"))
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
        return model, pipeline

    def _generate_response(self, messages: Union[List[Dict[str, str]], str]) -> Tuple[str, int, int, float, float]:
        args = PIPELINE_ARGS(
            temperature=self.config.get("temperature", 1.0),
            top_p=self.config.get("top_p", 0.7),
            alpha_frequency=self.config.get("alpha_frequency", 0.25),
            alpha_presence=self.config.get("alpha_presence", 0.25),
            token_ban=[],
            token_stop=[],
            chunk_len=self.config.get("chunk_len", 256),
        )

        if self.is_chat:
            prompt = self._generate_chat_prompt(messages)
        else:
            prompt = messages

        start_time = time.time()

        output = self.tokenizer.generate(prompt, token_count=self.max_new_tokens, args=args, callback=self._print_output)

        end_time = time.time()

        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(output))

        return output, input_tokens, output_tokens, start_time, end_time

    @staticmethod
    def _print_output(text):
        print(text, end="", flush=True)

    def _generate_chat_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"System: {message['content']}\n\n"
            elif message["role"] == "user":
                prompt += f"Human: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        prompt += "Assistant:"
        return prompt

