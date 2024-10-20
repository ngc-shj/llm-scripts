import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import List, Dict, Union
import time

class AIAssistant:
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.is_chat = (
            hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None
        )
        self.use_system_prompt = not args.no_use_system_prompt

    def _load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_path or self.args.model_path,
            trust_remote_code=True
        )
        # パディングトークンの設定
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        return model, tokenizer

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
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_new_tokens": self.args.max_tokens,
            "repetition_penalty": 1.1,
        }

        if self.is_chat:
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = messages

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        print("--- prompt")
        print(prompt)
        print("--- output")

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        start_time = time.time()

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
            **generation_params
        )
        end_time = time.time()

        output = self.tokenizer.decode(
            output_ids[0][input_ids.size(1):],
            skip_special_tokens=True
        )
        
        return output, input_ids, output_ids, start_time, end_time

    def _print_stats(self, input_ids, output_ids, start_time, end_time):
        input_tokens = len(input_ids[0])
        output_tokens = len(output_ids[0][input_ids.size(1):])
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
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = parse_arguments()
    assistant = AIAssistant(args)

    def q(
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None
    ) -> Union[List[Dict[str, str]], str]:
        return assistant.query(user_query, history)

    print('history = ""')
    print('history = q("ドラえもんとはなにか")')
    print('history = q("続きを教えてください", history)')

