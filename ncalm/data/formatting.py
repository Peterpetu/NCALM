"""Shared text rendering helpers for dataset tokenization and tokenizer training."""

from __future__ import annotations

from typing import Any


CHAT_SPECIAL_TOKENS = [
    "<|unk|>",
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|developer|>",
]


ROLE_TOKEN_MAP = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "developer": "<|developer|>",
}


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return text


def _render_tool_block(chat_template_kwargs: dict[str, Any] | None) -> str:
    if not chat_template_kwargs:
        return ""

    tool_lines: list[str] = []
    python_tools = chat_template_kwargs.get("python_tools") or []
    xml_tools = chat_template_kwargs.get("xml_tools") or []

    if python_tools:
        tool_lines.append("Python tools:")
        tool_lines.extend(f"- {tool}" for tool in python_tools)
    if xml_tools:
        tool_lines.append("XML tools:")
        tool_lines.extend(f"- {tool}" for tool in xml_tools)

    return "\n".join(tool_lines).strip()


def render_messages(messages: list[dict[str, Any]], chat_template_kwargs: dict[str, Any] | None = None) -> str:
    """Render a chat example into plain training text with explicit role tags."""
    parts: list[str] = []

    custom_instructions = ""
    if chat_template_kwargs:
        custom_instructions = _normalize_text(
            chat_template_kwargs.get("custom_instructions", "")
        )
    tool_block = _render_tool_block(chat_template_kwargs)

    system_parts = [block for block in [custom_instructions, tool_block] if block]
    if system_parts:
        parts.append("<|system|>")
        parts.append("\n\n".join(system_parts))

    for message in messages:
        role = _normalize_text(message.get("role", "user")).lower()
        content = _normalize_text(message.get("content", ""))
        if not content:
            continue

        role_token = ROLE_TOKEN_MAP.get(role, f"<|{role}|>")
        parts.append(role_token)
        parts.append(content)

    return "\n".join(parts).strip()


def example_to_text(example: dict[str, Any]) -> str:
    """Convert a dataset row into plain text suitable for tokenizer/model training."""
    if "text" in example and example["text"] is not None:
        return _normalize_text(example["text"])

    if "messages" in example and example["messages"] is not None:
        return render_messages(
            example["messages"],
            example.get("chat_template_kwargs"),
        )

    if "chosen" in example and example["chosen"] is not None:
        prompt = _normalize_text(example.get("prompt", ""))
        chosen_text = render_messages(example["chosen"])
        if prompt:
            return f"<|user|>\n{prompt}\n<|assistant|>\n{chosen_text}".strip()
        return chosen_text

    return ""
