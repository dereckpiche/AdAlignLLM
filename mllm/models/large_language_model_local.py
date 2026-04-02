"""
File: mllm/models/large_language_model_local.py
Summary: Provides a local large language model wrapper over inference backends.
"""

import logging
import os
import re
import sys
import uuid
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Literal

import httpx
import requests
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.chat_utils.apply_template import chat_turns_to_token_ids
from mllm.markov_games.rollout_tree import ChatTurn
from mllm.models.adapter_training_wrapper import AdapterWrapper
from mllm.models.inference_backend import LLMInferenceOutput
from mllm.models.inference_backend_dummy import DummyInferenceBackend
from mllm.models.inference_backend_vllm import VLLMAsyncBackend

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

AdapterID = str
PolicyID = str


class LeanLocalLLM:
    """
    Wrapper that manages local HuggingFace models, adapters, and inference backends.
    """

    def __init__(
        self,
        llm_id: str = "base_llm",
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        hf_kwargs: dict = {},
        adapter_configs: dict = {},
        output_directory: str = "./models/",
        inference_backend: Literal["vllm", "dummy"] = "vllm",
        inference_backend_sampling_params: dict = {},
        inference_backend_init_kwargs: dict = {},
        initial_adapter_paths: dict[str, str] | None = None,
        initial_buffer_paths: list[str] | None = None,
        enable_thinking: bool = None,
        regex_max_attempts: int = -1,
        max_thinking_characters: int = 0,
    ):
        self.inference_backend_name = inference_backend
        self.output_directory = output_directory
        self.llm_id = llm_id
        self.device = torch.device(device) if device else torch.device("cuda")
        self.model_name = model_name
        self.adapter_configs = adapter_configs
        self.adapter_ids = list(adapter_configs.keys())
        self.enable_thinking = enable_thinking
        self.regex_max_attempts = regex_max_attempts
        self.initial_buffer_paths = initial_buffer_paths
        self.max_thinking_characters = max_thinking_characters
        self.regex_retries_count = 0

        # Optional user-specified initial adapter weight locations (local or HF Hub)
        # Format: {adapter_id: path_or_repo_id}
        self.initial_adapter_paths: dict[str, str] | None = initial_adapter_paths

        # Path management / imports
        self.save_path = str(os.path.join(output_directory, model_name, "adapters"))
        self.adapter_paths = {
            adapter_id: os.path.join(self.save_path, adapter_id)
            for adapter_id in self.adapter_ids
        }
        checkpoints_dir = os.path.join(self.output_directory, "checkpoints")
        self.past_agent_adapter_paths = {}
        if os.path.isdir(checkpoints_dir):
            for dirname in os.listdir(checkpoints_dir):
                dirpath = os.path.join(checkpoints_dir, dirname)
                if os.path.isdir(dirpath):
                    self.past_agent_adapter_paths[f"{dirname}_buffer"] = os.path.join(
                        dirpath, "agent_adapter"
                    )
            logger.info(
                f"Loaded {len(self.past_agent_adapter_paths)} past agent adapters from checkpoints directory."
            )
        if self.initial_buffer_paths is not None:
            previous_count = len(self.past_agent_adapter_paths)
            for path in self.initial_buffer_paths:
                if os.path.isdir(path):
                    for dirname in os.listdir(path):
                        dirpath = os.path.join(path, dirname)
                        if os.path.isdir(dirpath):
                            self.past_agent_adapter_paths[
                                f"{dirname}_buffer"
                            ] = os.path.join(dirpath, "agent_adapter")
                else:
                    logger.warning(
                        f"Initial buffer path {path} does not exist or is not a directory."
                    )
            logger.info(
                f"Loaded {len(self.past_agent_adapter_paths) - previous_count} past agent adapters from user-specified initial buffer paths."
            )
        self.past_agent_adapter_ids = list(self.past_agent_adapter_paths.keys())

        # ID management for tracking adapter versions
        self.adapter_train_ids = {
            adapter_id: self.short_id_generator() for adapter_id in self.adapter_ids
        }
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Setup padding token to be same as EOS token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.weights_got_updated: dict[AdapterID, bool] = {
            adapter_id: False for adapter_id in self.adapter_ids
        }
        self.weights_got_updated.update(
            {adapter_id: False for adapter_id in self.past_agent_adapter_ids}
        )
        self.current_lora_request = None
        self.currently_loaded_adapter_id = None

        # ---------------------------------------------------------
        # Init HF model, peft adapters
        # ---------------------------------------------------------
        self.shared_hf_llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            **hf_kwargs,
        )
        self.hf_adapters = {}
        self.optimizers = {}
        for adapter_id in self.adapter_ids:
            # Prefer output-folder path if it exists; else fall back to user-specified initial path if provided
            output_path = os.path.join(self.save_path, adapter_id)
            chosen_path: str | None = None
            if os.path.isdir(output_path) and os.listdir(output_path):
                chosen_path = output_path
                logger.info(
                    f"Initializing adapter '{adapter_id}': using existing weights from output folder '{chosen_path}'."
                )
            elif (
                self.initial_adapter_paths and adapter_id in self.initial_adapter_paths
            ):
                chosen_path = self.initial_adapter_paths[adapter_id]
                logger.info(
                    f"Initializing adapter '{adapter_id}': using provided initial path '{chosen_path}'."
                )
            else:
                logger.info(
                    f"Initializing adapter '{adapter_id}': no initial weights provided or found; starting from scratch."
                )
            hf_adapter = AdapterWrapper(
                shared_llm=self.shared_hf_llm,
                adapter_id=adapter_id,
                lora_config=adapter_configs[adapter_id],
                path=chosen_path,
            ).to(device)
            self.hf_adapters[adapter_id] = hf_adapter
        # Persist current state of all adapters (ensures remote loads are cached to disk)
        self.export_adapters()

        # ---------------------------------------------------------
        # Init inference inference_backend
        # ---------------------------------------------------------

        if inference_backend == "vllm":
            self.inference_backend = VLLMAsyncBackend(
                model_name=self.model_name,
                # adapter_paths=self.adapter_paths,
                tokenizer=self.tokenizer,
                engine_init_kwargs=inference_backend_init_kwargs,
                sampling_params=inference_backend_sampling_params,
            )
        elif inference_backend == "dummy":
            self.inference_backend = DummyInferenceBackend()
        else:
            raise ValueError(f"Unknown inference_backend: {inference_backend}")

    def reset_regex_retries_count(self) -> None:
        self.regex_retries_count = 0

    def get_inference_policies(self) -> dict[PolicyID, Callable]:
        """
        Build async policy callables keyed by adapter id for inference-only usage.
        """
        policies = {}
        for adapter_id in self.adapter_ids:
            # define policy func
            async def policy(
                state: list[ChatTurn],
                agent_id: str,
                regex: str | None = None,
                _adapter_id=adapter_id,
            ):
                self.prepare_adapter_for_inference(adapter_id=_adapter_id)
                response = await self.get_action(state, agent_id, regex)
                return response

            policies[self.llm_id + "/" + adapter_id] = policy

        for adapter_id in self.past_agent_adapter_ids:
            # define policy func
            async def policy(
                state: list[ChatTurn],
                agent_id: str,
                regex: str | None = None,
                _adapter_id=adapter_id,
            ):
                self.prepare_adapter_for_inference(adapter_id=_adapter_id)
                response = await self.get_action(state, agent_id, regex)
                return response

            policies[self.llm_id + "/" + adapter_id] = policy
        return policies

    def get_adapter_modules(self) -> dict[PolicyID, nn.Module]:
        """
        Returns wrappers over the adapters which allows them be
        interfaced like regular PyTorch models.
        AdapterWrapper lives in adapter_wrapper.py; the huggingface modules already wrap
        parameters here, so we surface them directly until an extra shim is required.
        """
        trainable_objects = {an: self.hf_adapters[an] for an in self.adapter_ids}
        return trainable_objects

    async def toggle_training_mode(self) -> None:
        for adn in self.adapter_ids:
            self.adapter_train_ids[adn] = self.short_id_generator()
        await self.inference_backend.toggle_training_mode()

    async def toggle_eval_mode(self) -> None:
        await self.inference_backend.toggle_eval_mode()

    def prepare_adapter_for_inference(self, adapter_id: AdapterID) -> None:
        self.inference_backend.prepare_adapter(
            adapter_id,
            adapter_path=self.adapter_paths.get(
                adapter_id, self.past_agent_adapter_paths.get(adapter_id, None)
            ),
            weights_got_updated=self.weights_got_updated[adapter_id],
        )
        self.currently_loaded_adapter_id = adapter_id
        self.weights_got_updated[adapter_id] = False

    # def _make_prompt_text(self, prompt: list[dict]) -> str:
    #     if self.enable_thinking is not None:
    #         prompt_text = self.tokenizer.apply_chat_template(
    #             prompt,
    #             tokenize=False,
    #             add_generation_prompt=True,
    #             enable_thinking=self.enable_thinking,
    #         )
    #     else:
    #         prompt_text = self.tokenizer.apply_chat_template(
    #             prompt,
    #             tokenize=False,
    #             add_generation_prompt=True,
    #         )

    #     return prompt_text

    async def get_action(
        self, state: list[ChatTurn], agent_id: str, regex: str | None = None
    ) -> ChatTurn:
        current_regex = regex if self.regex_max_attempts == -1 else None
        pattern = re.compile(regex) if regex else None
        nb_attempts = 0
        state = state[:]
        while True:
            context_token_ids = chat_turns_to_token_ids(
                chats=state,
                tokenizer=self.tokenizer,
                enable_thinking=self.enable_thinking,
            )
            policy_output = await self.inference_backend.generate(
                input_token_ids=context_token_ids.tolist(),
                extract_thinking=(self.max_thinking_characters > 0),
                regex=current_regex,
            )
            if (
                pattern is None
                or (nb_attempts >= self.regex_max_attempts)
                or (pattern.fullmatch(policy_output.content))
            ):
                return ChatTurn(
                    agent_id=agent_id,
                    role="assistant",
                    content=policy_output.content,
                    reasoning_content=policy_output.reasoning_content,
                    out_token_ids=policy_output.out_token_ids,
                    log_probs=policy_output.log_probs,
                    is_state_end=False,
                )
            else:
                self.regex_retries_count += 1
                nb_attempts += 1
                logger.warning(
                    f"Response {policy_output.content} did not match regex: {regex}, retry {nb_attempts}/{self.regex_max_attempts}"
                )
                if nb_attempts == self.regex_max_attempts:
                    current_regex = regex
                # regex_prompt = ChatTurn(
                #     role="user",
                #     content=f"Invalid response format. Expected format (regex): {current_regex}\n Please try again and provide ONLY a response that matches this regex.",
                #     reasoning_content=None,
                #     log_probs=None,
                #     out_token_ids=None,
                #     is_state_end=False,
                # )
                # state.append(regex_prompt)

    def export_adapters(self) -> None:
        """
        Any peft wrapper, by default, saves all adapters, not just the one currently loaded.
        """

        # New version of the adapters available
        for adapter_id in self.adapter_ids:
            self.weights_got_updated[adapter_id] = True
        for adapter_id in self.past_agent_adapter_ids:
            self.weights_got_updated[adapter_id] = True

        adapter_id = self.adapter_ids[0]
        self.hf_adapters[adapter_id].save_pretrained(self.save_path)

    def checkpoint_all_adapters(self, checkpoint_indicator: str) -> None:
        """
        Checkpoints all adapters to the configured output directory.
        """
        adapter_id = self.adapter_ids[0]
        output_dir = os.path.join(self.output_directory, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        agent_adapter_dir = f"{adapter_id}-{checkpoint_indicator}-{date_str}"
        export_path = os.path.join(output_dir, agent_adapter_dir)
        for adapter_id in self.adapter_ids:
            if "agent" in adapter_id:
                self.past_agent_adapter_paths[
                    f"{agent_adapter_dir}_buffer"
                ] = os.path.join(export_path, adapter_id)
                self.past_agent_adapter_ids.append(f"{agent_adapter_dir}_buffer")
                self.weights_got_updated[f"{agent_adapter_dir}_buffer"] = False
                self.hf_adapters[adapter_id].save_pretrained(export_path)

    def short_id_generator(self) -> str:
        """
        Generates a short unique ID for tracking adapter versions.

        Returns:
            int: An 8-digit integer ID.
        """
        return str(uuid.uuid4().int)[:8]
