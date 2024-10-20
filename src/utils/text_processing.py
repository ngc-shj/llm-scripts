def clean_text(text: str) -> str:
    """Clean and normalize the input text."""
    return text.strip().replace('\r\n', '\n').replace('\n\n', '\n')

def truncate_text(text: str, max_length: int) -> str:
    """Truncate the text to the specified maximum length."""
    return text[:max_length]

def format_chat_message(role: str, content: str) -> str:
    """Format a chat message with role and content."""
    return f"{role}: {clean_text(content)}\n\n"

