import sys
import argparse
from mlx_lm import load, generate, stream_generate
from typing import List, Dict, Union
import time

class MLXAIAssistant:
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.is_chat = (
            hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None
        )
        self.use_system_prompt = not args.no_use_system_prompt

    def _load_model_and_tokenizer(self):
        return load(path_or_hf_repo=self.args.model_path)

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
            "temp": 0.8,
            "top_p": 0.95,
            "max_tokens": self.args.max_tokens,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
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

        print("--- prompt")
        print(prompt)
        print("--- output")

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
        
        return output, input_ids, output_ids, start_time, end_time

    def _print_stats(self, input_ids, output_ids, start_time, end_time):
        input_tokens = len(input_ids[0])
        output_tokens = len(output_ids[0])
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
        output, input_ids, output_ids, start_time, end_time = self._generate_response(messages)
        
        self._print_stats(input_ids, output_ids, start_time, end_time)

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
    parser.add_argument("--tokenizer-path", type=str, help="Path to the tokenizer (if different from model path)")
    parser.add_argument("--no-use-system-prompt", action='store_true', help="Do not use the default system prompt")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = parse_arguments()
    assistant = MLXAIAssistant(args)

    def q(
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None
    ) -> Union[List[Dict[str, str]], str]:
        return assistant.query(user_query, history)

    print('history = ""')
    print('history = q("ドラえもんとはなにか")')
    print('history = q("続きを教えてください", history)')

