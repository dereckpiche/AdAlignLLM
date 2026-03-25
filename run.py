"""
File: run.py
Summary: Main entry point for training and evaluation jobs.
"""

import asyncio
import copy
import importlib
import logging
import os
import pickle
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pandas.core.base import IndexLabel

import wandb
from mllm.markov_games.alternative_actions_runner import AlternativeActionsRunner
from mllm.markov_games.group_timesteps import group_by_round
from mllm.markov_games.linear_runner import LinearRunner
from mllm.markov_games.mg_utils import (
    AgentConfig,
    MarkovGameConfig,
    init_markov_game_components,
)
from mllm.markov_games.run_markov_games import run_markov_games
from mllm.models.human_policy import get_human_policies
from mllm.models.large_language_model_api import LargeLanguageModelOpenAI
from mllm.models.large_language_model_local import LeanLocalLLM
from mllm.models.scalar_critic import ScalarCritic
from mllm.training.trainer_ad_align import TrainerAdAlign
from mllm.training.trainer_independent import TrainerNaive
from mllm.training.trainer_sum_rewards import TrainerSumRewards
from mllm.utils.dict_get_path import get_from_nested_dict
from mllm.utils.resource_context import resource_logger_context
from mllm.utils.rollout_tree_stats import get_mean_rollout_tree_stats
from mllm.utils.short_id_gen import generate_short_id
from mllm.utils.update_start_epoch import update_start_epoch

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class ModulePointer:
    base_llm_id: str
    adapter_id: str


async def generate_and_train(cfg: dict, base_seed: int) -> None:
    """
    End-to-end experiment loop: generate rollouts, train adapters, log stats.

    ``cfg`` comes from Hydra (see configs/). ``base_seed`` lets you run multiple
    seeds from the same command line invocation.
    """
    # -----------------------------------------------------------------
    # Seed everything + resume randomness if random_state.pkl exists.
    # -----------------------------------------------------------------

    total_start_time = time.time()
    # Get Hydra's runtime output directory which includes date and config info.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = f"{hydra_cfg['runtime']['output_dir']}/seed_{base_seed}"
    os.makedirs(output_directory, exist_ok=True)

    update_start_epoch(cfg=cfg, output_directory=output_directory)

    random.seed(base_seed)  # Python random
    np.random.seed(base_seed)  # NumPy
    torch.manual_seed(base_seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(base_seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(base_seed)  # If using multi-GPU

    env_rng = np.random.default_rng(base_seed)

    random_state_dir = f"{output_directory}/random_state.pkl"
    # Load saved states
    wandb_run_id = str(generate_short_id())
    if os.path.exists(random_state_dir):
        with open(random_state_dir, "rb") as f:
            random_state_dict = pickle.load(f)
        print(f"Loaded random states from {random_state_dir}")
        random.setstate(random_state_dict["python"])
        np.random.set_state(random_state_dict["numpy"])
        torch.set_rng_state(random_state_dict["torch"])
        torch.cuda.set_rng_state_all(random_state_dict["torch_cuda"])
        wandb_run_id = random_state_dict.get("wandb_run_id", wandb_run_id)
    if cfg["experiment"].get("wandb_enabled", False):
        wandb.init(
            project="llm_negotiation",
            name=cfg["experiment"]["name"],
            config=cfg,
            resume="allow",
            id=wandb_run_id,
            mode=os.environ.get("WANDB_MODE", "online"),
        )

    # -----------------------------------------------------------------
    # Initialize models, critics, optimizers, trainers
    # -----------------------------------------------------------------

    # Step 1: instantiate every model defined in the Hydra config.
    llms_dict = {}
    for llm_id, model_config in cfg["models"].items():
        if model_config is None:
            continue
        model_class: LeanLocalLLM | LargeLanguageModelOpenAI = globals()[
            model_config["class"]
        ]  # Server-backed LLMs are temporarily disabled until the backend is rebuilt.
        llms_dict[llm_id] = model_class(
            **model_config["init_args"],
            output_directory=output_directory,
        )

    adapter_modules = {}  # Named references to trainable adapter modules.
    for llm_id, llm in llms_dict.items():
        if isinstance(llm, LeanLocalLLM):
            adapter_modules[llm_id] = llm.get_adapter_modules()

    # Scalar Critics
    critics = {}
    for critic_id, critic_config in cfg["critics"].items():
        if critic_config is None:
            continue
        critic_module_pointer = critic_config["module_pointer"]
        critic_adapter = get_from_nested_dict(adapter_modules, critic_module_pointer)
        critics[critic_id] = ScalarCritic(critic_adapter)

    trainable_modules = {**adapter_modules, **critics}

    # Init optimizers
    optimizers = {}
    for optimizer_id, optimizer_config in cfg["optimizers"].items():
        if optimizer_config is None:
            continue
        optimizer_module_pointer = optimizer_config["module_pointer"]
        module = get_from_nested_dict(trainable_modules, optimizer_module_pointer)
        optimizer_class: torch.optim.Adam | torch.optim.SGD = eval(
            optimizer_config["optimizer_class_name"]
        )
        init_args = optimizer_config["init_args"]
        optimizers[optimizer_id] = optimizer_class(module.parameters(), **init_args)

    # Step 2: wire up trainers (AdAlign / Naive / SumRewards) with correct modules.
    trainers = {}
    for trainer_id, trainer_config in cfg["trainers"].items():
        if trainer_config is None:
            continue
        trainer_class = eval(trainer_config["class"])
        module_pointers = trainer_config["module_pointers"]
        tokenizer = llms_dict[module_pointers["policy"][0]].tokenizer
        policy = get_from_nested_dict(adapter_modules, module_pointers["policy"])
        policy_optimizer = get_from_nested_dict(
            optimizers, module_pointers["policy_optimizer"]
        )
        if module_pointers.get("critic", False):
            critic = get_from_nested_dict(critics, module_pointers["critic"])
        else:
            critic = None
        if module_pointers.get("critic_optimizer", False):
            critic_optimizer = get_from_nested_dict(
                optimizers, module_pointers["critic_optimizer"]
            )
        else:
            critic_optimizer = None
        trainer: TrainerAdAlign | TrainerNaive | TrainerSumRewards = trainer_class(
            policy=policy,
            policy_optimizer=policy_optimizer,
            critic=critic,
            critic_optimizer=critic_optimizer,
            tokenizer=tokenizer,
            lr_scheduler=None,  # Learning-rate schedulers can plug in once configs supply them.
            critic_lr_scheduler=None,  # Critic schedulers share the same gap and remain unset.
            save_path=os.path.join(output_directory, trainer_id),
            **trainer_config["kwargs"],
        )
        trainers[trainer_id] = trainer

    # Stuff common across iterations
    agent_configs = []
    for agent_config_ in cfg["markov_games"]["agents"].values():
        agent_config = AgentConfig(**agent_config_)
        agent_configs.append(agent_config)

    nb_matches = cfg["experiment"]["nb_matches_per_iteration"]
    seed_group_size = cfg["experiment"].get("seed_group_size", 1)
    assert (
        nb_matches % seed_group_size == 0
    ), "nb_matches must be divisible by seed_group_size"

    # If the simulation supports random episode length via continuation probability,
    # we sample the round count here so trainers see a consistent length per iteration.
    if cfg["markov_games"].get("continuation_probability", False):
        cfg["markov_games"]["simulation_init_args"]["nb_of_rounds"] = min(
            (
                env_rng.geometric(
                    1 - cfg["markov_games"].get("continuation_probability", 0.85)
                )
            ),
            16,
        )
    for iteration in range(
        cfg["experiment"]["start_epoch"], cfg["experiment"]["nb_epochs"]
    ):
        logger.info(f"Starting iteration {iteration}.")
        # -----------------------------------------------------------------
        # Iteration loop: generate fresh trajectories
        # -----------------------------------------------------------------
        for llm in llms_dict.values():
            await llm.toggle_eval_mode()

        # Set folders and seeds
        env_rng = np.random.default_rng(env_rng.integers(0, 1e9))
        crn_rng = copy.deepcopy(env_rng)  # used for common-random-number seeds
        iteration_start_time = time.time()
        it_folder = os.path.join(output_directory, f"iteration_{iteration:03}")
        crn_seeds = [
            crn_rng.integers(0, 1e9, 1)[0] for _ in range(nb_matches // seed_group_size)
        ]  # common random number seeds
        os.makedirs(it_folder, exist_ok=True)

        # Get dictionnary of functionnal-like callable policies (only for inference)
        policies = {}
        for llm_id, llm in llms_dict.items():
            policies.update(llm.get_inference_policies())
        # Add human-in-the-loop policy
        policies.update(get_human_policies())

        # Optionally mix in buffer policies (past checkpoints or hard-coded agents).
        policy_ids = list(policies.keys())
        buffer_policy_ids = [
            policy_id for policy_id in policy_ids if "buffer" in policy_id
        ]
        human_policy_ids = [
            policy_id for policy_id in policy_ids if "human" in policy_id
        ]
        regular_policy_ids_length = (
            len(policy_ids) - len(buffer_policy_ids) - len(human_policy_ids)
        )
        logger.info(
            f"Inference policies count is regular policies {regular_policy_ids_length} and buffer policies {len(buffer_policy_ids)} and human policies {len(human_policy_ids)}."
        )
        logger.info(
            f"Hard coded buffer agents are set to {cfg['markov_games'].get('hard_coded_buffer_agents', False)} with prob {cfg['experiment'].get('prob_hard_coded_buffer_agent', 0)}"
        )
        if (
            cfg["experiment"].get("agent_buffer_recent_k", -1) != -1
            and len(buffer_policy_ids) > 0
        ):
            buffer_policy_ids = sorted(
                buffer_policy_ids,
                key=lambda x: int(re.search(r"iter_(\d+)", x).group(1)),
                reverse=True,
            )[: cfg["experiment"]["agent_buffer_recent_k"]]
        env_rng = np.random.default_rng(env_rng.integers(0, 1e9))
        buffer_rng = copy.deepcopy(env_rng)
        subsample_size = (
            cfg["experiment"].get("keep_agent_buffer_count", 10)
            - regular_policy_ids_length
        )
        if len(buffer_policy_ids) > subsample_size:
            buffer_policy_ids = buffer_rng.choice(
                buffer_policy_ids,
                size=subsample_size,
                replace=False,
            ).tolist()

        generation_start_time = time.time()

        # Create new markov games
        markov_games = []
        agent_ids = set()
        agent_ids.update([agent_config.agent_id for agent_config in agent_configs])
        buffer_networks_are_available = len(buffer_policy_ids) > 0
        buffer_hard_coded_agents_are_available = cfg["markov_games"].get(
            "hard_coded_buffer_agents", False
        )
        buffer_agents_are_available = (
            buffer_networks_are_available or buffer_hard_coded_agents_are_available
        )
        opp_agent_log_ids = {}
        for match_number in range(nb_matches):

            def agent_configs_per_match(agent_configs, match_number, group_number, id):
                take_buffer_agent = (
                    buffer_agents_are_available
                    and buffer_rng.random()
                    < cfg["markov_games"].get("buffer_prob", 0.5)
                )
                new_agent_configs = []
                for index, agent_config in enumerate(agent_configs):
                    if (group_number % len(agent_configs)) == index:
                        new_agent_configs.append(agent_config)
                    elif take_buffer_agent:
                        if not buffer_networks_are_available:
                            use_hard_coded = True
                        else:
                            use_hard_coded = buffer_rng.random() < cfg[
                                "experiment"
                            ].get("prob_hard_coded_buffer_agent", 0)
                        # use hard coded buffer agent
                        if use_hard_coded:
                            hc_buffer_config = copy.deepcopy(
                                buffer_rng.choice(
                                    list(
                                        cfg["markov_games"][
                                            "hard_coded_buffer_agents"
                                        ].values()
                                    )
                                )
                            )
                            hc_agent_config = copy.deepcopy(agent_config)
                            hc_agent_config.agent_id = f"{agent_config.agent_id}_buffer"
                            hc_agent_config.agent_class_name = hc_buffer_config[
                                "agent_class_name"
                            ]
                            agent_ids.add(hc_agent_config.agent_id)
                            new_agent_configs.append(hc_agent_config)

                        # use buffer network
                        else:
                            opp_agent_config = copy.deepcopy(agent_config)
                            opp_agent_config.agent_id = (
                                f"{agent_config.agent_id}_buffer"
                            )
                            opp_agent_log_ids[id] = f"{opp_agent_config.agent_id}"
                            agent_ids.add(opp_agent_config.agent_id)
                            buffer_agent_policy_ids = [
                                policy_id
                                for policy_id in buffer_policy_ids
                                if agent_config.policy_id in policy_id
                            ]
                            opp_agent_config.policy_id = buffer_rng.choice(
                                buffer_agent_policy_ids
                            )
                            new_agent_configs.append(opp_agent_config)
                    else:
                        opp_agent_config = copy.deepcopy(agent_config)
                        opp_agent_config.agent_id = f"{agent_config.agent_id}_buffer"
                        opp_agent_log_ids[id] = f"{opp_agent_config.agent_id}_live"
                        agent_ids.add(opp_agent_config.agent_id)
                        new_agent_configs.append(opp_agent_config)
                return new_agent_configs

            markov_game_config = MarkovGameConfig(
                id=iteration * nb_matches + match_number,
                seed=int(crn_seeds[match_number // seed_group_size]),
                simulation_class_name=cfg["markov_games"]["simulation_class_name"],
                simulation_init_args=cfg["markov_games"]["simulation_init_args"],
                agent_configs=agent_configs_per_match(
                    agent_configs,
                    match_number,
                    match_number // seed_group_size,
                    iteration * nb_matches + match_number,
                )
                if cfg["experiment"].get("agent_buffer", False)
                else agent_configs,
            )
            markov_game = init_markov_game_components(
                config=markov_game_config, policies=policies
            )
            markov_games.append(markov_game)

        # Generate rollouts raw data asynchronously (LinearRunner or Alternative runner).
        runner = eval(cfg["markov_games"]["runner_method_name"])
        rollout_trees = await run_markov_games(
            runner=runner,
            runner_kwargs=cfg["markov_games"]["runner_kwargs"],
            output_folder=it_folder,
            markov_games=markov_games,
        )

        # This will merge all timesteps of a round into a single timestep - simplifies credit assignment during training
        if cfg["markov_games"].get("group_by_round", False):
            rollout_trees = [
                group_by_round(rollout_tree) for rollout_tree in rollout_trees
            ]

        log_rollout_trees = []
        # Export rollout trees
        for i, rollout_tree in enumerate(rollout_trees):
            with open(
                os.path.join(it_folder, f"mgid_{rollout_tree.id}.rt.pkl"), "wb"
            ) as f:
                log_rollout_tree = copy.deepcopy(rollout_tree)
                for idx, agent_id in enumerate(log_rollout_tree.agent_ids):
                    if "buffer" in agent_id:
                        log_rollout_tree.agent_ids[idx] = opp_agent_log_ids[
                            log_rollout_tree.id
                        ]
                # Update all agent IDs in the rollout tree
                node = log_rollout_tree.child
                while node is not None:
                    # Update agent IDs in simulation step log
                    if (
                        hasattr(node.step_log, "simulation_step_log")
                        and node.step_log.simulation_step_log
                    ):
                        sim_log = node.step_log.simulation_step_log

                        # Update action_logs keys and chat turns agent_id
                        action_logs = node.step_log.action_logs
                        for a_id, act_log in action_logs.items():
                            if "buffer" in a_id:
                                mapped_id = opp_agent_log_ids[log_rollout_tree.id]
                            else:
                                mapped_id = a_id

                            # Update chat turns' agent_id to the mapped id
                            for chat_turn in act_log.chat_turns:
                                if "buffer" in chat_turn.agent_id:
                                    chat_turn.agent_id = mapped_id

                        # Update rewards dictionary
                        if hasattr(sim_log, "rewards") and sim_log.rewards:
                            new_rewards = {}
                            for agent_id, reward in sim_log.rewards.items():
                                if "buffer" in agent_id:
                                    new_agent_id = opp_agent_log_ids[
                                        log_rollout_tree.id
                                    ]
                                    new_rewards[new_agent_id] = reward
                                else:
                                    new_rewards[agent_id] = reward
                            sim_log.rewards = new_rewards

                        # Update info dictionary - values and splits
                        if hasattr(sim_log, "info") and sim_log.info:
                            info = sim_log.info

                            # Update values dictionary
                            if "values" in info:
                                new_values = {}
                                for agent_id, value in info["values"].items():
                                    if "buffer" in agent_id:
                                        new_agent_id = opp_agent_log_ids[
                                            log_rollout_tree.id
                                        ]
                                        new_values[new_agent_id] = value
                                    else:
                                        new_values[agent_id] = value
                                info["values"] = new_values

                            # Update splits dictionary
                            if "splits" in info:
                                new_splits = {}
                                for agent_id, split in info["splits"].items():
                                    if "buffer" in agent_id:
                                        new_agent_id = opp_agent_log_ids[
                                            log_rollout_tree.id
                                        ]
                                        new_splits[new_agent_id] = split
                                    else:
                                        new_splits[agent_id] = split
                                info["splits"] = new_splits

                    node = node.child

                # Store as pure Python dict to avoid class dependency on load
                pickle.dump(
                    log_rollout_tree.model_dump(), f, protocol=pickle.HIGHEST_PROTOCOL
                )
                log_rollout_trees.append(log_rollout_tree)

        # Optionally compute live rollout stats for wandb streaming.
        live_wandb_rollout_tree_stats_module = cfg["experiment"].get(
            "stat_methods_for_live_wandb", False
        )
        generation_metrics = {}
        if live_wandb_rollout_tree_stats_module:
            try:
                mod = importlib.import_module(live_wandb_rollout_tree_stats_module)
                metrics: list[Callable[[Any], List[Tuple[str, float]]]] = []
                # Module must have a stat_functs list attribute
                metrics = getattr(mod, "stat_functs", None)
                generation_metrics = get_mean_rollout_tree_stats(
                    log_rollout_trees, metrics
                ).data
            except Exception as e:
                logger.error(f"Error computing live stats for wandb: {e}")

        # Add number of regex retries to generation metrics
        total_iteration_regex_retries = 0
        for llm in llms_dict.values():
            if hasattr(llm, "regex_retries_count"):
                total_iteration_regex_retries += llm.regex_retries_count
                llm.reset_regex_retries_count()
        generation_metrics["regex_retries_count"] = total_iteration_regex_retries
        logger.info(
            f"Number of regex retries in iteration {iteration}: {total_iteration_regex_retries}"
        )

        logger.info(
            f"agents played in iteration {iteration} are {', '.join(list(agent_ids))}"
        )
        generation_end_time = time.time()

        # Process raw data into training data using the specified functions for each agent

        # -----------------------------------------------------------------
        # Training phase: switch adapters to train mode, run trainer loops
        # -----------------------------------------------------------------
        if not cfg["experiment"]["train"]:
            if cfg["experiment"].get("wandb_enabled", False):
                wandb.log(generation_metrics, step=iteration)
            continue

        training_start_time = time.time()

        # Prepare base models for training
        for llm in llms_dict.values():
            await llm.toggle_training_mode()

        # ----------- Training (with advantage sharing between trainers)
        # Send advantage packets to other trainers
        all_advantage_packets = []
        for trainer_id, trainer in trainers.items():
            for agent_id in cfg["train_on_which_data"][trainer_id]:
                agent_id_roots = [
                    root for root in rollout_trees if agent_id in root.agent_ids
                ]
                trainer.set_agent_trajectory_data(
                    agent_id=agent_id,
                    roots=agent_id_roots,
                )
                buffer_agent_id = f"{agent_id}_buffer"
                if buffer_agent_id in agent_ids:
                    buffer_agent_id_roots = [
                        root
                        for root in rollout_trees
                        if buffer_agent_id in root.agent_ids
                    ]
                    trainer.set_agent_trajectory_data(
                        agent_id=buffer_agent_id,
                        roots=buffer_agent_id_roots,
                    )

            # trainer.set_trajectory_data(
            #     rollout_trees=rollout_trees,
            #     agent_ids=cfg["train_on_which_data"][trainer_id],
            # )
            advantage_packets = trainer.share_advantage_data()
            all_advantage_packets.extend(advantage_packets)

        # Receive advantage packets from other trainers and train
        trainer_metrics = {}
        for trainer_id, trainer in trainers.items():
            trainer.receive_advantage_data(all_advantage_packets)
            trainer.set_policy_gradient_data(
                agent_ids=cfg["train_on_which_data"][trainer_id]
            )
            trainer_metrics[trainer_id] = trainer.train()

        # Persist per-trainer logs + optimizer/annealing states so the next iteration can resume.
        for trainer_id, trainer in trainers.items():
            trainer.export_training_tally(
                identifier=trainer_id,
                folder=it_folder,
            )
            trainer.export_trainer_states()

        training_end_time = time.time()

        # Periodically checkpoint adapters (so vLLM inference can pick up weights mid-run).
        checkpoint_frequency = cfg["experiment"]["checkpoint_every_n_iterations"]
        if (
            checkpoint_frequency != -1
            and iteration % checkpoint_frequency == 0
            and iteration != 0
        ):
            for llm_id, llm in llms_dict.items():
                if hasattr(llm, "adapter_paths"):
                    llm.checkpoint_all_adapters(
                        checkpoint_indicator=f"iter_{iteration}"
                    )
        # Log iteration-level metrics (wall-clock + trainer stats) to wandb.
        iteration_metrics = {}
        iteration_metrics.update(generation_metrics)
        for trainer_id, metrics in trainer_metrics.items():
            for key, value in metrics.items():
                iteration_metrics[f"{trainer_id}/{key}"] = value
        if cfg["experiment"].get("wandb_enabled", False):
            wandb.log(iteration_metrics, step=iteration)

        # Export all HF adapters weights (needed for vLLM inference)
        for llm in llms_dict.values():
            llm.export_adapters()
        iteration_end_time = time.time()

        # Timing calculations
        iteration_duration = iteration_end_time - iteration_start_time
        generation_duration = generation_end_time - generation_start_time
        training_duration = training_end_time - training_start_time

        generation_percentage = (generation_duration / iteration_duration) * 100
        training_percentage = (training_duration / iteration_duration) * 100

        elapsed_time = iteration_end_time - total_start_time
        estimated_total_time = iteration_duration * cfg["experiment"]["nb_epochs"]
        estimated_remaining_time = estimated_total_time - elapsed_time

        time_per_iteration = iteration_duration
        time_est_10 = time_per_iteration * 10
        time_est_100 = time_per_iteration * 100
        time_est_500 = time_per_iteration * 500

        def format_time(seconds):
            if seconds >= 3600:
                return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
            elif seconds >= 60:
                return f"{int(seconds // 60)}m {int(seconds % 60)}s"
            else:
                return f"{int(seconds)}s"

        logger.info(
            f"Iteration {iteration + 1} took {format_time(iteration_duration)} "
            f"({generation_percentage:.2f}% Gen, {training_percentage:.2f}% Train). "
            f"Generation: {format_time(generation_duration)}, "
            f"Training: {format_time(training_duration)}. "
            f"Estimated remaining time: {format_time(estimated_remaining_time)}. "
            f"Estimated total time: {format_time(estimated_total_time)}. "
            f"Time estimates for 10 more iterations: {format_time(time_est_10)}, "
            f"100 more iterations: {format_time(time_est_100)}, "
            f"500 more iterations: {format_time(time_est_500)}."
        )

        # Snapshot RNG states so resuming a run reproduces the exact same data.
        random_state_dict = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "wandb_run_id": wandb_run_id,
        }
        with open(random_state_dir, "wb") as f:
            pickle.dump(random_state_dict, f)


@hydra.main(config_path="./configs")
def main(cfg):
    # Hydra handles CLI parsing; this wrapper just sets up logging & kicks off `generate_and_train`.
    hydra_run_dir = HydraConfig.get().run.dir
    filename = os.path.join(hydra_run_dir, "generate_and_train_log.log")
    logging.basicConfig(filename=filename, level=logging.INFO)

    # Output source code in runtime directory for certain reproducibility
    os.makedirs(hydra_run_dir, exist_ok=True)
    shutil.copytree(
        "mllm",
        os.path.join(hydra_run_dir, "src_code_for_reproducibility"),
        dirs_exist_ok=True,
    )
    # Run the experiment specified in the configuration
    try:
        asyncio.run(
            generate_and_train(
                OmegaConf.to_container(
                    cfg, resolve=True, structured_config_mode="dict"
                ),
                base_seed=cfg.experiment.base_seed,
            )
        )
    finally:
        if cfg.experiment.get("wandb_enabled", False):
            wandb.finish()
        # Clean up distributed process groups if they exist
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
