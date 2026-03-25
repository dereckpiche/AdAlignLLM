# Learning Robust Social Strategies with Large Language Models

This repository accompanies [*Learning Robust Social Strategies with Large Language Models*](https://arxiv.org/pdf/2511.19405v1) and provides everything needed to reproduce its experiments. It enables you to fine-tune LLMs with RL, Advantage Alignment, and pit heterogeneous LLM agents against each other in social dilemmas.

## Key Features

- **Multi-agent Markov games**: Iterated Prisoner's Dilemma, Trust & Split variants.
- **Async multi-turn rollouts**: Agents act concurrently via `asyncio` with hot-swappable LoRA adapters on shared base models (vLLM).
- **Analysis tools**: Renderers and statistics collectors for qualitative evaluation.

---

## Installation

Recommended: **Python ≥ 3.11** and **CUDA ≥ 12.4**.

```bash
git clone https://github.com/dereckpiche/AdAlignLLM.git
cd AdAlignLLM
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install vllm==0.10.1.1
pip install triton==3.3.1
pip install transformers==4.57.1
pip install flash-attn==2.8.3 --no-deps --no-build-isolation --no-binary flash-attn
pip install -r requirements.txt
pip install -e .
```


For OpenAI API models:
```bash
export OPENAI_API_KEY=your_api_key
```

## Development Setup

```bash
pip install pre-commit
pre-commit install
pip install nbstripout
nbstripout --install
export WANDB_PROJECT=llm_negotiation
```

## Running Experiments

Results are saved to `$SCRATCH/llm_negotiation/`. Set the environment variable if needed:
```bash
export SCRATCH=/path/to/scratch/directory
```

### Training
Launch policy gradient training:
```bash
python run.py --config-name tas_rps_vanilla_ad_align.yaml
```

### Rendering Results
Generate visualizations:
```bash
python render.py --nego /path/to/experiment
```

## Evaluation

Download trained adapters from [HuggingFace](https://huggingface.co/LLMnegotiation) and update config paths accordingly.

### Human vs AI
```bash
python run.py --config-name human_bob_tas_rps.yaml
python render.py --nego /path/to/experiment --html
```

### Benchmarking
```bash
python run_benchmarks.py --config-name benchmark_tas_rps.yaml
python render_benchmarks.py /path/to/benchmark/results
```

## Adding a Markov Game environment

1. **Implement a `Simulation` subclass.** Use `mllm/markov_games/simulation.py` as the contract: your environment must define `step`, `get_obs_agent`, safe-copy/reset helpers, and any optional state/action accessors you plan to log. Place the implementation under `mllm/markov_games/<your_env>/` (see `negotiation/` or `ipd/` for templates).
2. **Create matching `Agent` wrappers.** Derive from `mllm/markov_games/agent.py` to describe how observations are converted into prompts/policies for each participant. These wrappers encapsulate tokenization, policy calls, and structured action parsing for your environment.
3. **Wire everything through `MarkovGame` configs.** Update or create config objects that feed into `mllm/markov_games/mg_utils.py`. Each `MarkovGameConfig` ties together agent IDs, agent classes, and the simulation class so runners like `run_markov_games.py` or the training scripts can instantiate your environment from Hydra configs.
4. **Expose the environment to training/eval scripts.** Reference the new config in `run.py`/Hydra configs or extend helpers like `statistics_runner.py` if you need custom logging. Once the simulation and agents satisfy the abstractions in `markov_game.py`, your environment becomes plug-and-play across trainers, renderers, and evaluation utilities.
