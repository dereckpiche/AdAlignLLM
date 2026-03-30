"""
File: mllm/models/large_language_model_api.py
Summary: Implements API-based large-language-model inference adapters.
"""

from __future__ import annotations

import asyncio
import copy
import os
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import backoff
from openai import AsyncOpenAI, OpenAIError

from mllm.markov_games.rollout_tree import ChatTurn
from mllm.models.inference_backend import LLMInferenceOutput

# Static list copied from the public OpenAI docs until a discovery endpoint is exposed.
reasoning_models = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "o1-mini",
    "o1",
    "o1-pro",
    "o3-mini",
    "o3",
    "o3-pro",
    "o4-mini",
    "o4",
    "o4-pro",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


class LargeLanguageModelOpenAI:
    """Tiny async wrapper for OpenAI Chat Completions."""

    def __init__(
        self,
        llm_id: str = "",
        model: str = "gpt-4.1-mini",
        reasoning_effort: str = "low",
        add_constraint_msg: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 300.0,
        regex_max_attempts: int = 10,
        sampling_params: Optional[Dict[str, Any]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        output_directory: Optional[str] = None,
    ) -> None:
        self.llm_id = llm_id
        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Set OPENAI_API_KEY as global environment variable or pass api_key."
            )
        client_kwargs: Dict[str, Any] = {"api_key": key, "timeout": timeout_s}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)

        # Sampling/default request params set at init
        self.sampling_params = sampling_params
        self.use_reasoning = model in reasoning_models
        if self.use_reasoning:
            self.sampling_params["reasoning"] = {
                "effort": reasoning_effort,
                "summary": "detailed",
            }
        self.regex_max_attempts = max(1, int(regex_max_attempts))
        self.add_constraint_msg = add_constraint_msg

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

    def extract_output_from_response(self, resp: Response) -> LLMInferenceOutput:
        if len(resp.output) > 1:
            reasoning_content = resp.output[0].content
            summary = resp.output[0].summary
            if reasoning_content is not None:
                reasoning_content = (
                    f"OpenAI Reasoning Content: {reasoning_content[0].text}"
                )
            elif summary != []:
                reasoning_content = f"OpenAI Reasoning Summary: {summary[0].text}"
            else:
                reasoning_content = None
            content = resp.output[1].content[0].text
        else:
            reasoning_content = None
            content = resp.output[0].content[0].text

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
        # Remove any non-role/content keys from the prompt else openai will error.
        prompt = [{"role": p.role, "content": p.content} for p in state]

        # if self.sleep_between_requests:
        #     await self.wait_random_time()

        # If regex is required, prime the model and validate client-side
        if regex:
            if self.add_constraint_msg:
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
                resp = await self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    **self.sampling_params,
                )
                policy_output = self.extract_output_from_response(resp)
                if pattern.fullmatch(policy_output.content):
                    return policy_output
                prompt = [
                    *prompt,
                    {
                        "role": "user",
                        "content": (
                            f"Invalid response format. Expected format (regex): {regex}\n Please try again and provide ONLY a response that matches this regex."
                        ),
                    },
                ]
            return policy_output

        # Simple, unconstrained generation
        resp = await self.client.responses.create(
            model=self.model,
            input=prompt,
            **self.sampling_params,
        )
        policy_output = self.extract_output_from_response(resp)
        return policy_output

    def shutdown(self) -> None:
        self.client = None
