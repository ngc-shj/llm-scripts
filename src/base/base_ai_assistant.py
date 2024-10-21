from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple
import time

class BaseAIAssistant(ABC):
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.is_chat = self._determine_chat_mode()
        self.use_system_prompt = not args.no_use_system_prompt
        self.max_new_tokens = self._get_max_tokens()
        self.system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    @abstractmethod
    def _load_model_and_tokenizer(self):
        pass

    def _determine_chat_mode(self):
        return hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None

    def _get_max_tokens(self):
        return min(self.args.max_tokens, self._get_model_max_tokens())

    def _get_model_max_tokens(self):
        return 4096  # Default value, override in subclasses if necessary

    def _prepare_messages(
        self,
        user_query: str,
        history: Union[List[Dict[str, str]], str] = None
    ) -> Union[List[Dict[str, str]], str]:
        if self.is_chat:
            messages = (
                [{"role": "system", "content": self.system_prompt}]
                if self.use_system_prompt
                else []
            )
            messages.extend(history or [])
            messages.append({"role": "user", "content": user_query})
        else:
            system_prompt = f"{self.system_prompt}\n\n" if self.use_system_prompt else ""
            messages = f"{system_prompt}{history or ''}{user_query}"
        return messages

    @abstractmethod
    def _generate_response(
        self,
        messages: Union[List[Dict[str, str]], str]
    ) -> Tuple[str, int, int, float, float]:
        pass

    def _print_stats(self, input_tokens: int, output_tokens: int, start_time: float, end_time: float):
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
        output, input_tokens, output_tokens, start_time, end_time = self._generate_response(messages)
        
        self._print_stats(input_tokens, output_tokens, start_time, end_time)

        if self.is_chat:
            return (history or []) + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": output}
            ]
        else:
            return f"{history or ''}{user_query}{output}"

