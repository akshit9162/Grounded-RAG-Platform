"""
LLM client. Defaults to the Anthropic API (Claude), but is written behind a
narrow interface so swapping to a self-hosted open-weight model served via
vLLM/TGI is a one-class change, not a rewrite — useful if you later need to
control inference cost/latency directly instead of paying per-token.
"""
from __future__ import annotations

import os
from typing import Protocol

from rag_platform.config import settings


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str: ...


class AnthropicLLMClient:
    def __init__(self, model: str):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class VLLMClient:
    """
    OpenAI-compatible client for a self-hosted vLLM/TGI server. Use this in
    production if you're serving an open-weight model yourself instead of
    calling a hosted API. Point VLLM_BASE_URL at your inference server.
    """

    def __init__(self, model: str, base_url: str):
        from openai import OpenAI  # optional dep; vLLM exposes an OpenAI-compatible API

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.getenv("VLLM_API_KEY", "not-needed"))

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class MockExtractiveLLMClient:
    """
    Zero-dependency, zero-API-key stand-in for a real LLM call. It does NOT
    generate free-form text — it extracts the most query-relevant sentences
    directly from the excerpts it's given and stitches them together with
    citations, mimicking the shape of a grounded answer.

    This exists purely so the pipeline is runnable end-to-end (e.g. in a
    sandboxed CI environment or before you've provisioned an API key)
    without silently pretending to be a real LLM. It is deliberately
    obvious in its own output ("[extractive]") so nobody mistakes a demo
    run for a real generation quality result — swap in AnthropicLLMClient
    or VLLMClient for anything you'd actually report eval numbers on.
    """

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        import re

        from rag_platform.text_utils import meaningful_tokens

        # Pull the excerpts back out of the user_prompt (built by
        # generation/prompts.py) so we can extract from them directly.
        blocks = re.findall(r"\[(\d+)\] \(source: [^)]*\)\n(.*?)(?=\n\n\[\d+\]|\n\nAnswer|$)", user_prompt, re.S)
        question_match = re.search(r"Question: (.*)", user_prompt)
        question = question_match.group(1).strip() if question_match else ""
        question_tokens = meaningful_tokens(question)

        scored_sentences = []
        for idx_str, block_text in blocks:
            idx = int(idx_str)
            for sentence in re.split(r"(?<=[.!?])\s+", block_text.strip()):
                tokens = meaningful_tokens(sentence)
                overlap = len(tokens & question_tokens)
                # Require at least two shared meaningful (non-stopword)
                # tokens — a single incidental word in common with a long
                # policy sentence isn't enough signal to call it relevant.
                if overlap >= 2:
                    scored_sentences.append((overlap, idx, sentence.strip()))

        if not scored_sentences:
            return "I don't have enough information in the available sources to answer that."

        scored_sentences.sort(key=lambda x: -x[0])
        top = scored_sentences[:3]
        answer_parts = [f"{sentence} [{idx}]" for _, idx, sentence in top]
        return "[extractive] " + " ".join(answer_parts)


_llm_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton

    if settings.llm_provider == "vllm":
        _llm_singleton = VLLMClient(settings.llm_model, os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    elif settings.llm_provider == "mock" or not os.getenv("ANTHROPIC_API_KEY"):
        _llm_singleton = MockExtractiveLLMClient()
    else:
        _llm_singleton = AnthropicLLMClient(settings.llm_model)
    return _llm_singleton
