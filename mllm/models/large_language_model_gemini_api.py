"""
File: mllm/models/large_language_model_gemini_api.py
Summary: Implements native Gemini API-based large-language-model inference adapters.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Callable, Dict, List, Optional

import backoff
from google import genai
from google.genai import types

from mllm.markov_games.rollout_tree import ChatTurn
from mllm.models.inference_backend import LLMInferenceOutput


class LargeLanguageModelGemini:
    """Tiny async wrapper for the native Gemini API."""

    def __init__(
        self,
        llm_id: str = "",
        model: str = "gemini-3.1-flash-lite-preview",
        api_key: Optional[str] = None,
        timeout_s: float = 300.0,
        regex_max_attempts: int = 10,
        sampling_params: Optional[Dict[str, Any]] = None,
        thinking_level: str = "low",
        include_thoughts: bool = True,
        init_kwargs: Optional[Dict[str, Any]] = None,
        output_directory: Optional[str] = None,
    ) -> None:
        self.llm_id = llm_id
        self.model = model
        self.timeout_s = timeout_s
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Set GEMINI_API_KEY as global environment variable or pass api_key."
            )
        self.client = genai.Client(api_key=key)
        self.sampling_params = sampling_params or {}
        self.thinking_level = thinking_level
        self.include_thoughts = include_thoughts
        self.regex_max_attempts = max(1, int(regex_max_attempts))

    def get_inference_policies(self) -> Dict[str, Callable]:
        return {
            self.llm_id: self.get_action,
        }

    async def prepare_adapter_for_inference(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def toggle_eval_mode(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def toggle_training_mode(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def export_adapters(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    async def checkpoint_all_adapters(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        pass

    def messages_to_contents(self, messages: List[Dict[str, str]]) -> List[types.Content]:
        contents: List[types.Content] = []
        system_chunks: List[str] = []

        for message in messages:
            role = message["role"]
            text = message["content"]

            if role == "system":
                system_chunks.append(text)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=text)],
                )
            )

        if system_chunks:
            system_text = "\n\n".join(system_chunks)
            contents.insert(
                0,
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "System instruction:\n"
                                f"{system_text}\n\n"
                                "Follow the system instruction for the rest of this conversation."
                            )
                        )
                    ],
                ),
            )

        return contents

    def build_generate_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=self.thinking_level,
                include_thoughts=self.include_thoughts,
            ),
            **self.sampling_params,
        )

    def extract_output_from_response(self, response: Any) -> LLMInferenceOutput:
        reasoning_parts: List[str] = []
        content_parts: List[str] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    reasoning_parts.append(text)
                else:
                    content_parts.append(text)

        content = "\n".join(content_parts) if content_parts else (response.text or "")
        reasoning_content = "\n".join(reasoning_parts) if reasoning_parts else None

        return LLMInferenceOutput(
            content=content,
            reasoning_content=reasoning_content,
        )

    @backoff.on_exception(
        backoff.expo, Exception, max_time=10**10, max_tries=10**10
    )
    async def get_action(
        self,
        state: list[ChatTurn],
        agent_id: str,
        regex: Optional[str] = None,
    ) -> LLMInferenceOutput:
        prompt = [{"role": p.role, "content": p.content} for p in state]

        if regex:
            constraint_msg = {
                "role": "user",
                "content": (
                    f"Output must match this regex exactly: {regex} \n"
                    "Return only the matching string, with no quotes or extra text."
                ),
            }
            prompt = [constraint_msg, *prompt]
            pattern = re.compile(regex)
            for _ in range(self.regex_max_attempts):
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=self.messages_to_contents(prompt),
                    config=self.build_generate_config(),
                )
                policy_output = self.extract_output_from_response(response)
                if pattern.fullmatch(policy_output.content):
                    return policy_output
                prompt = [
                    *prompt,
                    {
                        "role": "user",
                        "content": (
                            f"Invalid response format. Expected format (regex): {regex}\n"
                            "Please try again and provide ONLY a response that matches this regex."
                        ),
                    },
                ]
            return policy_output

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=self.messages_to_contents(prompt),
            config=self.build_generate_config(),
        )
        return self.extract_output_from_response(response)

    def shutdown(self) -> None:
        self.client = None
