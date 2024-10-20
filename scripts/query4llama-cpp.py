import sys
import os
import argparse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, llama_chat_format
from typing import List, Dict, Union
import time

class LlamaAIAssistant:
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

    def __init__(self, args):
        self.args = args
        self.model, self.chat_formatter, self.chat_handler = self._load_model_and_handlers()
        self.is_chat = not args.no_chat
        self.use_system_prompt = not args.no_use_system_prompt

    def _load_model_and_handlers(self):
        local_model_path = os.path.join(self.args.ggml_model_path, self.args.ggml_model_file)
        if os.path.isfile(local_model_path):
            ggml_model_path = local_model_path
        else:
            ggml_model_path = hf_hub_download(
                self.args.ggml_model_path,
                filename=self.args.ggml_model_file
            )

        chat_formatter = llama_chat_format.hf_autotokenizer_to_chat_formatter(self.args.model_path)
        chat_handler = llama_chat_format.hf_autotokenizer_to_chat_completion_handler(self.args.model_path)

        model = Llama(
            model_path=ggml_model_path,
            chat_handler=chat_handler,
            n_ctx=self.args.n_ctx,
            n_threads=self.args.n_threads,
            n_gpu_layers=self.args.n_gpu_layers
        )

        return model, chat_formatter, chat_handler

    def _prepare_messages(
        self,
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None
    ) -> Union[List[Dict[str, str]], str]:
        if self.is_chat:
            messages = (
                [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}] if self.use_system_prompt else []
            )
            messages.extend(history or [])
            messages.append({"role": "user", "content": user_query})
        else:
            system_prompt = (
                f"{self.DEFAULT_SYSTEM_PROMPT}\n\n" if self.use_system_prompt else ""
            )
            messages = f"{system_prompt}{history or ''}{user_query}"
        return messages

    def _generate_response(self, messages: Union[List[Dict[str, str]], str]):
        generation_params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": self.args.max_tokens,
            "repeat_penalty": 1.1,
        }

        if self.is_chat:
            prompt = self.chat_formatter(messages=messages)
        else:
            prompt = messages

        print("--- messages")
        print(messages)
        print("--- prompt")
        print(prompt)
        print("--- output")

        start_time = time.time()

        if self.is_chat:
            outputs = self.model.create_chat_completion(
                messages=messages,
                **generation_params
            )
            output = outputs["choices"][0]["message"]["content"]
        else:
            outputs = self.model.create_completion(
                prompt=prompt,
                **generation_params
            )
            output = outputs["choices"][0]["text"]

        end_time = time.time()

        print(output)

        return output, outputs, start_time, end_time

    def _print_stats(self, outputs, start_time, end_time):
        input_tokens = outputs["usage"]["prompt_tokens"]
        output_tokens = outputs["usage"]["completion_tokens"]
        total_time = end_time - start_time
        tps = output_tokens / total_time
        print(f"prompt tokens = {input_tokens:.7g}")
        print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
        print(f"   total time = {total_time:f} [s]")

    def query(
        self,
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None
    ) -> Union[List[Dict[str, str]], str]:
        messages = self._prepare_messages(user_query, history)
        output, outputs, start_time, end_time = self._generate_response(messages)

        self._print_stats(outputs, start_time, end_time)

        if self.is_chat:
            return (history or []) + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": output}
            ]
        else:
            return f"{history or ''}{user_query}{output}"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--ggml-model-path", type=str, required=True, help="Path to the GGML model directory")
    parser.add_argument("--ggml-model-file", type=str, required=True, help="GGML model file name")
    parser.add_argument("--no-chat", action='store_true', help="Disable chat mode")
    parser.add_argument("--no-use-system-prompt", action='store_true', help="Do not use the default system prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    parser.add_argument("--n-threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of GPU layers to use")
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = parse_arguments()
    assistant = LlamaAIAssistant(args)

    def q(
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None
    ) -> Union[List[Dict[str, str]], str]:
        return assistant.query(user_query, history)

    print('history = ""')
    print('history = q("ドラえもんとはなにか")')
    print('history = q("続きを教えてください", history)')

