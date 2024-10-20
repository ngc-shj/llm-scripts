import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import RWKVAIAssistant
from src.utils import parse_arguments
from src.config import get_config

def main():
    args = parse_arguments()
    config = get_config("rwkv")
    assistant = RWKVAIAssistant(args, config)

    print('Welcome to the RWKV AI Assistant!')
    print('Type "exit" to end the conversation.')

    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = assistant.query(user_input, history)
        if isinstance(response, list):
            history = response
            #print(f"Assistant: {response[-1]['content']}")
        else:
            history += user_input + response
            #print(f"Assistant: {response}")

if __name__ == "__main__":
    main()

