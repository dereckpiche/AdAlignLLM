"""
File: mllm/models/inference_backend_vllm.py
Summary: Connects to in-process vLLM instances for batched generation.
"""

import asyncio
import re
from typing import Optional

import torch
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams, RequestOutputKind

from mllm.models.inference_backend import LLMInferenceBackend, LLMInferenceOutput
from mllm.utils.short_id_gen import generate_short_id


class VLLMAsyncBackend(LLMInferenceBackend):
    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        # adapter_paths: dict[str, str],
        engine_init_kwargs: dict = {},
        sampling_params: dict = {},
    ):
        self.model_name = model_name
        self.vllm_adapter_ids = {}
        ea = dict(model=model_name, **engine_init_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**ea))

        self.sampling_params = sampling_params
        self.tokenizer = tokenizer

    def prepare_adapter(
        self,
        adapter_id: Optional[str],
        adapter_path: Optional[str],
        weights_got_updated: bool,
    ) -> None:
        if weights_got_updated:
            self.vllm_adapter_ids[adapter_id] = generate_short_id()
        self.current_lora_request = LoRARequest(
            adapter_id,
            self.vllm_adapter_ids[adapter_id],
            adapter_path,
        )

    async def toggle_training_mode(self) -> None:
        await self.engine.sleep(level=1)

    async def toggle_eval_mode(self) -> None:
        await self.engine.wake_up()

    def shutdown(self) -> None:
        # No explicit close call; engine stops when process exits.
        pass

    async def generate(
        self,
        input_token_ids: list[int],
        regex: Optional[str] = None,
        extract_thinking: bool = False,
    ) -> LLMInferenceOutput:
        # Build SamplingParams correctly
        guided = GuidedDecodingParams(regex=regex) if regex else None
        sp = SamplingParams(
            **self.sampling_params,
            guided_decoding=guided,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        prompt = TokensPrompt(prompt_token_ids=input_token_ids)
        request_id = f"req-{asyncio.get_running_loop().time()}"
        result_generator = self.engine.generate(
            prompt,
            sp,  # SamplingParams(...)
            request_id,
            lora_request=self.current_lora_request,
        )

        async for out in result_generator:  # with FINAL_ONLY this runs once
            res = out

        raw_text = res.outputs[0].text
        out_token_ids = res.outputs[0].token_ids
        log_probs = [
            logprob_dict[token_id].logprob
            for token_id, logprob_dict in zip(out_token_ids, res.outputs[0].logprobs)
        ]
        log_probs = torch.tensor(log_probs)
        out_token_ids = torch.tensor(out_token_ids, dtype=torch.long)
        content = raw_text
        reasoning_content = None

        if extract_thinking:
            m = re.match(
                r"^\n<think>\n([\s\S]*?)</think>\n\n(.*)$", raw_text, flags=re.DOTALL
            )
            if m:
                reasoning_content = m.group(1)
                content = m.group(2)
        return LLMInferenceOutput(
            content=content,
            reasoning_content=reasoning_content,
            log_probs=log_probs,
            out_token_ids=out_token_ids,
        )
