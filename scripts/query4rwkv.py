import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import sys
import argparse
import torch
from huggingface_hub import hf_hub_download
from typing import List, Dict, Union
import time

class RWKVAIAssistant:
    DEFAULT_SYSTEM_PROMPT = "わたしは誠実で優秀な日本人のアシスタントです。"

    def __init__(self, args):
        self.args = args
        self.model, self.pipeline = self._load_model_and_pipeline()
        self.is_chat = not args.no_chat
        self.use_system_prompt = not args.no_use_system_prompt
        self.max_new_tokens = min(3500, args.max_tokens)
        self.pipeline_args = PIPELINE_ARGS(
            temperature=1.0,
            top_p=0.85,
            top_k=0,
            alpha_frequency=0.2,
            alpha_presence=0.2,
            alpha_decay=0.996,
            token_ban=[],
            token_stop=[],
            chunk_len=256
        )

    @staticmethod
    def print_nolf(outstr):
        print(outstr, end="", flush=True)

    def _load_model_and_pipeline(self):
        local_model_path = os.path.join(self.args.model_path, self.args.model_file)
        if os.path.isfile(local_model_path):
            model_path = local_model_path
        else:
            model_path = hf_hub_download(self.args.model_path, filename=self.args.model_file)

        model = RWKV(model=model_path, strategy='cuda fp16')
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
        return model, pipeline

    def _generate_chat_prompt(self, conversation: List[Dict[str, str]], add_generation_prompt=True) -> str:
        prompt = ""
        for message in conversation:
            role = message["role"]
            content = message["content"].strip().replace('\r\n','\n').replace('\n\n','\n')
            if message["role"] == "system":
                prompt += f"User: こんにちは\n\nAssistant: {content}\n\n"
            else:
                prompt += f"{role}: {content}\n\n"
        if add_generation_prompt:
            prompt += "Assistant:"
        return prompt

    def _generate_prompt(self, user_query: str, instruction: str=None, add_generation_prompt=True) -> str:
        prompt = ""
        if instruction:
            prompt += f"Instruction: {instruction}\n\n"
        prompt += f"Input: {user_query}\n\n"
        if add_generation_prompt:
            prompt += f"Output:"
        return prompt

    def _prepare_messages(self, user_query: str, history: Union[List[Dict[str, str]], str] = None) -> Union[List[Dict[str, str]], str]:
        if self.is_chat:
            messages = (
                [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
                if self.use_system_prompt
                else []
            )
            messages.extend(history or [])
            messages.append({"role": "User", "content": user_query})
        else:
            messages = f"{history or ''}{user_query}"
        return messages

    def _generate_response(self, messages: Union[List[Dict[str, str]], str], instruction: str = None):
        if self.is_chat:
            prompt = self._generate_chat_prompt(messages, add_generation_prompt=True)
        else:
            prompt = self._generate_prompt(messages, instruction, add_generation_prompt=True)

        print("--- prompt")
        print(prompt)
        print("--- output")

        start_time = time.time()

        output = self.pipeline.generate(
            ctx=prompt,
            token_count=self.max_new_tokens,
            args=self.pipeline_args,
            callback=self.print_nolf
        )

        end_time = time.time()
        
        return output, prompt, start_time, end_time

    def _print_stats(self, prompt, output, start_time, end_time):
        input_ids = self.pipeline.encode(prompt)
        input_tokens = len(input_ids)
        output_ids = self.pipeline.encode(output)
        output_tokens = len(output_ids)
        total_time = end_time - start_time
        tps = output_tokens / total_time
        print("\n---")
        print(f"prompt tokens = {input_tokens:.7g}")
        print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
        print(f"   total time = {total_time:f} [s]")

    def query(
        self,
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None,
        instruction: str = None
    ) -> Union[List[Dict[str, str]], str]:
        messages = self._prepare_messages(user_query, history)
        output, prompt, start_time, end_time = self._generate_response(messages, instruction)
        
        self._print_stats(prompt, output, start_time, end_time)

        if self.is_chat:
            return (history or []) + [
                {"role": "User", "content": user_query},
                {"role": "Assistant", "content": output}
            ]
        else:
            return f"{history or ''}{user_query}{output}"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-file", type=str, required=True, help="Model file name")
    parser.add_argument("--no-chat", action='store_true', help="Disable chat mode")
    parser.add_argument("--no-use-system-prompt", action='store_true', help="Do not use the default system prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = parse_arguments()
    assistant = RWKVAIAssistant(args)

    def q(
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None,
        instruction: str = None
    ) -> Union[List[Dict[str, str]], str]:
        return assistant.query(user_query, history, instruction)

    print('history = ""')
    print('history = q("ドラえもんとはなにか")')
    print('history = q("続きを教えてください", history)')

