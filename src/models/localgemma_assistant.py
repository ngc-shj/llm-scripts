import torch
from transformers import AutoTokenizer, TextStreamer
from local_gemma import LocalGemma2ForCausalLM
from src.base import BaseAIAssistant
from src.utils.torch_utils import get_torch_dtype
from typing import List, Dict, Union, Tuple
import time

class LocalGemmaAIAssistant(BaseAIAssistant):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.use_system_prompt=False

    def _load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_path or self.args.model_path,
            trust_remote_code=True
        )
        
        model = LocalGemma2ForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=get_torch_dtype(self.config.get("torch_dtype", "float32")),
            device_map=self.config.get("device_map", "auto"),
            preset=self.config.get("preset", "auto"),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # Ensure tokenizer settings align with model configuration
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.model_max_length > model.config.max_position_embeddings:
            tokenizer.model_max_length = model.config.max_position_embeddings

        return model, tokenizer

    def _get_model_max_tokens(self):
        return self.model.config.max_position_embeddings

    def _generate_response(self, messages: Union[List[Dict[str, str]], str]) -> Tuple[str, int, int, float, float]:
        generation_params = {
            "do_sample": True,
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 0.9),
            "top_k": self.config.get("top_k", 40),
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.config.get("repetition_penalty", 1.1),
        }

        if self.is_chat:
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = messages

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        start_time = time.time()

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
            **generation_params
        )

        end_time = time.time()

        output = self.tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)
        
        input_tokens = len(input_ids[0])
        output_tokens = len(output_ids[0]) - input_tokens

        return output, input_tokens, output_tokens, start_time, end_time

