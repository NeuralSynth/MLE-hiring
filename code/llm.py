import os
import re
import json
from pathlib import Path
from openai import OpenAI

# Load .env so this module works whether imported via main.py or standalone
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # dotenv not installed; rely on environment variables being set externally


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.model = os.getenv("LLM_MODEL")  # must match `ollama list` output exactly
        self.client = self._init_client()

    def _init_client(self):
        if self.provider == "ollama":
            return OpenAI(
                base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("LOCAL_LLM_KEY", "ollama"),  # required by openai lib, value ignored by ollama
            )
        elif self.provider == "groq":
            from groq import Groq
            return Groq(api_key=os.getenv("GROQ_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def complete(self, system: str, user: str) -> str:
        if self.provider in ("ollama", "groq", "openai"):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content.strip()
        elif self.provider == "anthropic":
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text.strip()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced {...} span, ignoring braces inside strings,
    or None if there is no balanced object. Lets us tolerate preamble/suffix
    prose around the JSON (e.g. "Sure! {...}") that weaker models add."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def clean_json_response(text: str) -> str:
    """Strip thinking tags and Markdown fences, then isolate the JSON object."""
    # Remove <think>...</think> blocks (Qwen3 chain-of-thought)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # If wrapped in a Markdown code fence, take its contents
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    text = text.strip()
    # Tolerate preamble/suffix prose around the object
    extracted = _extract_json_object(text)
    return extracted if extracted is not None else text


llm = LLMClient()  # singleton — import this everywhere
