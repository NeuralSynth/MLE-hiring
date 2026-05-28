import os
import re
import json
import hashlib
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
        elif self.provider == "gemini":
            from google import genai
            return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def complete(self, system: str, user: str) -> str:
        if self.provider in ("ollama", "groq", "openai"):
            # Prompt caching is automatic for the OpenAI-shaped providers:
            #   - openai: caches prefixes >=1024 tokens on gpt-4o / gpt-4o-mini
            #     / o-series. Passing prompt_cache_key (a stable hash of the
            #     system prompt) gives the load balancer a consistent routing
            #     key so cache hits don't bounce across backend servers.
            #   - groq: server-side automatic prefix caching on supported
            #     models; no API parameter to tune.
            #   - ollama: llama.cpp KV cache reuses the system prefix as long
            #     as the model stays loaded; no API knob in OpenAI-compat.
            # prompt_cache_key is OpenAI-only — it's sent via extra_body so
            # this stays compatible with older openai SDK versions, and only
            # for the openai provider since Groq/Ollama may reject unknown
            # body fields.
            kwargs = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if self.provider == "openai":
                kwargs["extra_body"] = {"prompt_cache_key": _cache_key_for(system)}
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        elif self.provider == "anthropic":
            # Prompt caching: the five system prompts (safety, classifier,
            # escalation, generator-normal, generator-escalate) are static
            # across every ticket, so we mark the system block with
            # cache_control so Anthropic serves it from the prompt cache after
            # the first call (lower TTFT, ~90% lower cost on cached tokens).
            # If the prompt is below the model's minimum cacheable size,
            # cache_control is silently ignored — harmless. Other providers
            # cache implicitly (Ollama KV cache) or automatically (OpenAI /
            # Groq server-side prefix cache), so no change is needed there.
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text.strip()
        elif self.provider == "gemini":
            # Prompt caching: Gemini supports implicit caching on 2.5+ models
            # (Flash / Pro) for prefixes above the model's minimum size — no
            # API parameter required, the platform detects and reuses cached
            # prefixes automatically. Explicit caching via client.caches.create
            # exists for heavier reuse patterns but is overkill here.
            response = self.client.models.generate_content(
                model=self.model,
                contents=user,
                config={
                    "system_instruction": system,
                    "temperature": 0,
                },
            )
            return (response.text or "").strip()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


def _cache_key_for(system: str) -> str:
    """Stable 16-char hex key derived from the system prompt. Two calls with
    the same system prompt share a key, which helps OpenAI route them to the
    same backend so the cached prefix is hit reliably."""
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


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
