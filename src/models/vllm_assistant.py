from vllm import LLM, SamplingParams
from src.base import BaseAIAssistant
from typing import List, Dict, Union, Tuple
import time

class VLLMAIAssistant(BaseAIAssistant):
    def __init__(self, args, config):
        super().__init__(args, config)

    def _load_model_and_tokenizer(self):
        model = LLM(
            model=self.args.model_path,
            dtype="auto",
            tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.7)
        )
        tokenizer = model.get_tokenizer()
        return model, tokenizer

    def _generate_response(self, messages: Union[List[Dict[str, str]], str]) -> Tuple[str, int, int, float, float]:
        sampling_params = SamplingParams(
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
            top_k=self.config.get("top_k", 40),
            max_tokens=self.max_new_tokens,
        )

        if self.is_chat:
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = messages

        start_time = time.time()

        outputs = self.model.generate(
            prompt_token_ids=self.tokenizer.encode(prompt),
            sampling_params=sampling_params
        )

        end_time = time.time()

        output = outputs[0].outputs[0].text
        input_tokens = len(outputs[0].prompt_token_ids)
        output_tokens = len(outputs[0].outputs[0].token_ids)

        return output, input_tokens, output_tokens, start_time, end_time

