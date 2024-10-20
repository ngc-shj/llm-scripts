import sys
import argparse
from vllm import LLM, SamplingParams
from typing import List, Dict, Union
import time

class VLLMAIAssistant:
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

    def __init__(self, args):
        self.args = args
        self.model = self._load_model()
        self.tokenizer = self.model.get_tokenizer()
        self.is_chat = (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        )
        self.use_system_prompt = not args.no_use_system_prompt

    def _load_model(self):
        return LLM(
            model=self.args.model_path,
            dtype="auto",
            trust_remote_code=True,
            tensor_parallel_size=self.args.tensor_parallel_size,
            max_model_len=self.args.max_model_len,
            gpu_memory_utilization=self.args.gpu_memory_utilization
        )

    def _prepare_messages(
        self,
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None
    ) -> Union[List[Dict[str, str]], str]:
        if self.is_chat:
            messages = (
                [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
                if self.use_system_prompt
                else []
            )
            messages.extend(history or [])
            messages.append({"role": "user", "content": user_query})
        else:
            system_prompt = (
                f"{self.DEFAULT_SYSTEM_PROMPT}\n\n"
                if self.use_system_prompt
                else ""
            )
            messages = f"{system_prompt}{history or ''}{user_query}"
        return messages

    def _generate_response(self, messages: Union[List[Dict[str, str]], str]):
        generation_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            max_tokens=self.args.max_tokens,
            repetition_penalty=1.1
        )

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
        )

        print("--- prompt")
        print(prompt)
        print("--- output")

        start_time = time.time()

        outputs = self.model.generate(
            sampling_params=generation_params,
            prompt_token_ids=[input_ids],
        )
        output = outputs[0]
        print(output.outputs[0].text)

        end_time = time.time()
        
        return output, start_time, end_time

    def _print_stats(self, output, start_time, end_time):
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
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
        output, start_time, end_time = self._generate_response(messages)
        
        self._print_stats(output, start_time, end_time)

        if self.is_chat:
            return (history or []) + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": output.outputs[0].text}
            ]
        else:
            return f"{history or ''}{user_query}{output.outputs[0].text}"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--no-use-system-prompt", action='store_true', help="Do not use the default system prompt")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Maximum model length")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2, help="GPU memory utilization")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = parse_arguments()
    assistant = VLLMAIAssistant(args)

    def q(
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None
    ) -> Union[List[Dict[str, str]], str]:
        return assistant.query(user_query, history)

    print('history = ""')
    print('history = q("ドラえもんとはなにか")')
    print('history = q("続きを教えてください", history)')

