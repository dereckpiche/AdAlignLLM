# Learning Robust Social Strategies with Large Language Models

This repository accompanies [*Learning Robust Social Strategies with Large Language Models*](https://arxiv.org/pdf/2511.19405v1) and provides everything needed to reproduce its experiments. It lets you pit heterogeneous LLM agents against each other in social dilemmas, fine-tune them with Advantage Alignment, and inspect how negotiation transcripts evolve round by round. The code is built around async multi-turn agents that share a base model and swap LoRA adapters per agent or policy.

If you are new here, think of the repo as:

- A collection of Markov game environments (Iterated Prisoner's Dilemma, Trust & Split variants, negotiation tasks, etc.) implemented with a consistent simulator and agent abstraction.
- A LoRA-first multi-agent orchestration layer that launches async multi-turn generations across many agents or policies (see `run_markov_games.py` and the `LinearRunner`), hot-swapping per-agent LoRA adapters on a shared base model via vLLM.
- A training stack that turns conversation logs into trajectories, applies multiple credit assignment methods, and fine-tunes LLM policies via RL or supervised objectives.
- Runners and analytics scripts (renderers, statistics collectors) so you can quickly iterate on new environments or policy variants.

---

## Key capabilities

- **Async multi-agent multi-turn rollouts**: Agents act concurrently at each timestep via `asyncio`, and multiple Markov games can run in parallel. vLLM’s async engine plus `LoRARequest` support lets you hot-swap adapters per agent without reloading the base model.
- **LoRA training for many agents**: `AdapterWrapper` registers one adapter per agent on a shared HF model. `TrainerAdAlign`, `TrainerNaive`, and `TrainerSumRewards` then optimize the corresponding adapters.

---

## Installation

Recommended: **Python ≥ 3.11** and **CUDA ≥ 12.4**.

```bash
git clone https://github.com/dereckpiche/AdAlignLLM.git
cd AdAlignLLM

pip install -r requirements.txt
pip install flash_attn --no-build-isolation
pip install -e .


and run

```python
import sys
sys.path.append('your-path-to-the-repo')
```
in order to add the repository to your system path.

To run with OpenAI API models, set
```bash
export OPENAI_API_KEY=your/api/key
```

## Development

```bash
pip install pre-commit
pre-commit install
pip install nbstripout
nbstripout --install
export WANDB_PROJECT=llm_negotiation
```

## Running Experiments

In order to launch a policy gradient training loop, use
```bash
python run.py --config-name your-config
```
Example: `python run.py --config-name tas_rps_vanilla_ad_align.yaml`.

To add render files to your output folder, use
```bash
python render.py --simulation_name path
```
Example: `python render.py --nego your-path-to-the-experiment`.

To play against a trained model in Trust & Split RPS, run
```bash
python run.py --config-name human_bob_tas_rps.yaml
```
Ensure the agent adapter path in that config point to your downloaded weights from https://huggingface.co/LLMnegotiation.

## Adding a Markov Game environment

1. **Implement a `Simulation` subclass.** Use `mllm/markov_games/simulation.py` as the contract: your environment must define `step`, `get_obs_agent`, safe-copy/reset helpers, and any optional state/action accessors you plan to log. Place the implementation under `mllm/markov_games/<your_env>/` (see `negotiation/` or `ipd/` for templates).
2. **Create matching `Agent` wrappers.** Derive from `mllm/markov_games/agent.py` to describe how observations are converted into prompts/policies for each participant. These wrappers encapsulate tokenization, policy calls, and structured action parsing for your environment.
3. **Wire everything through `MarkovGame` configs.** Update or create config objects that feed into `mllm/markov_games/mg_utils.py`. Each `MarkovGameConfig` ties together agent IDs, agent classes, and the simulation class so runners like `run_markov_games.py` or the training scripts can instantiate your environment from Hydra configs.
4. **Expose the environment to training/eval scripts.** Reference the new config in `run.py`/Hydra configs or extend helpers like `statistics_runner.py` if you need custom logging. Once the simulation and agents satisfy the abstractions in `markov_game.py`, your environment becomes plug-and-play across trainers, renderers, and evaluation utilities.
