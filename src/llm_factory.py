from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Type, TypeVar

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv(override=True)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def _build_gemini_llm() -> ChatGoogleGenerativeAI | None:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.1,
        google_api_key=api_key,
    )


def _build_groq_llm() -> ChatGroq | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        return None
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    return ChatGroq(
        model=model_name,
        temperature=0.1,
        api_key=api_key,
    )


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse JSON from a model response, stripping markdown fences if present."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected top-level JSON object from LLM.")
    return parsed


def invoke_structured_with_failover(
    system_prompt: str,
    compact_payload: Dict[str, Any],
    schema: Type[SchemaT],
) -> tuple[SchemaT, str]:
    """
    Invoke structured output with automatic failover across Gemini and Groq.
    Returns (validated_pydantic_instance, provider_model_label).
    """
    load_dotenv(override=True)
    compact_json = json.dumps(compact_payload, separators=(",", ":"), ensure_ascii=True)
    user_content = f"INPUT_JSON:{compact_json}"

    providers: List[tuple[str, Any]] = []
    gemini_llm = _build_gemini_llm()
    groq_llm = _build_groq_llm()
    if gemini_llm is not None:
        providers.append((f"gemini:{gemini_llm.model}", gemini_llm))
    if groq_llm is not None:
        providers.append((f"groq:{groq_llm.model}", groq_llm))

    if not providers:
        raise RuntimeError(
            "No valid LLM API key configured. Set GOOGLE_API_KEY or GROQ_API_KEY in .env."
        )

    errors: List[str] = []
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    for label, llm in providers:
        try:
            structured_llm = llm.with_structured_output(schema)
            result = structured_llm.invoke(messages)
            if isinstance(result, schema):
                return result, label
            if isinstance(result, dict):
                return schema.model_validate(result), label
        except Exception as exc_structured:
            # Secondary JSON-mode fallback for strict TPM/schema models
            try:
                raw_resp = llm.invoke(messages)
                raw_text = (
                    raw_resp.content
                    if isinstance(raw_resp.content, str)
                    else json.dumps(raw_resp.content, separators=(",", ":"))
                )
                parsed_obj = _extract_json_object(raw_text)
                validated = schema.model_validate(parsed_obj)
                return validated, f"{label} (json-fallback)"
            except Exception as exc_raw:
                errors.append(f"{label}: structured={exc_structured}; raw={exc_raw}")

    raise RuntimeError(f"All LLM providers failed structured output: {' | '.join(errors)}")