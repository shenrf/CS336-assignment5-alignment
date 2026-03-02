# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CS336 Spring 2025 Assignment 5 on alignment techniques: Supervised Fine-Tuning (SFT), Expert Iteration, and GRPO (Group Relative Policy Optimization) with verified rewards on MATH. An optional supplement covers safety alignment, instruction tuning, and RLHF (DPO).

## Setup

```bash
uv sync --no-install-package flash-attn
uv sync
```

The model fixture used in tests expects `Qwen2.5-Math-1.5B` at `/data/a5-alignment/models/Qwen2.5-Math-1.5B`.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_sft.py

# Run a single test by name
uv run pytest tests/test_sft.py::test_compute_entropy

# Run tests verbosely
uv run pytest -v

# Run tests with exact snapshot matching (stricter tolerances)
uv run pytest --snapshot-exact

# Create submission zip
bash test_and_make_submission.sh
```

## Architecture

### Adapter Pattern

All implementation work happens through `tests/adapters.py`. Students implement functions elsewhere and wire them in here. Tests never import directly from `cs336_alignment` — they call the adapter functions. **To connect your implementation to the tests, complete the `raise NotImplementedError` stubs in `tests/adapters.py`.**

### Core Package: `cs336_alignment/`

- `drgrpo_grader.py` — Math answer grading for GRPO reward signal. Provides high-recall grading via `math_verify`, `sympy`, and LaTeX parsing. This is the reward function for MATH problem solving.
- `prompts/` — Text prompt files used across experiments:
  - `r1_zero.prompt` — DeepSeek-R1-Zero style prompt
  - `question_only.prompt` — Bare question prompt
  - `zero_shot_system_prompt.prompt` — Zero-shot system prompt
  - `alpaca_sft.prompt` — AlpacaEval SFT format prompt

### Tests: `tests/`

- `adapters.py` — **The main file to edit.** Contains all function stubs students must implement.
- `conftest.py` — Shared pytest fixtures (model, tokenizer, tensors, reward functions).
- `_snapshots/` — Pre-computed `.npz` (numpy) and `.pkl` snapshots for numerical tests. Tests compare against these using `rtol=1e-4, atol=1e-2` by default.
- `fixtures/` — Small models and data for testing: `tiny-gpt2`, `tiny-gpt2-ref`, `Meta-Llama-3-8B` tokenizer, `sft_sample.jsonl`.

### Test Files by Topic

| File | Topic |
|------|-------|
| `test_sft.py` | Tokenization, entropy, log-probs, masked normalize, SFT train step |
| `test_grpo.py` | Group reward normalization, policy gradient losses, masked mean, GRPO train step |
| `test_data.py` | Packed SFT dataset construction, batch iteration (optional/RLHF) |
| `test_metrics.py` | MMLU and GSM8K response parsing (optional/RLHF) |
| `test_dpo.py` | DPO per-instance loss (optional/RLHF) |

### Data: `data/`

Evaluation datasets: `alpaca_eval/`, `gsm8k/`, `mmlu/`, `simple_safety_tests/`.

### Scripts: `scripts/`

- `evaluate_safety.py` — Runs safety evaluation pipeline
- `alpaca_eval_vllm_llama3_3_70b_fn/` — AlpacaEval judge configuration using Llama 3.3 70B via vLLM

## Key Implementation Functions

The adapters correspond to these algorithmic components:

- **`run_tokenize_prompt_and_output`** — Tokenize prompt+response pairs and build response mask (1 for response tokens, 0 for prompt/padding). Returns `input_ids`, `labels` (shifted), `response_mask`.
- **`run_masked_mean` / `run_masked_normalize`** — Aggregation utilities that respect a binary mask.
- **`run_compute_entropy`** — Token entropy from logits.
- **`run_get_response_log_probs`** — Forward pass returning per-token log-probs and optionally entropy.
- **`run_compute_group_normalized_rewards`** — GRPO advantage computation: groups rollouts, normalizes within each group. The `reward_fn` callable has signature `(response: str, ground_truth: str) -> dict` with keys `"reward"`, `"format_reward"`, `"answer_reward"`.
- **`run_compute_naive_policy_gradient_loss`** / **`run_compute_grpo_clip_loss`** — Per-token policy gradient losses.
- **`run_compute_policy_gradient_loss`** — Dispatcher that delegates to the appropriate loss based on `loss_type` (`"no_baseline"`, `"reinforce_with_baseline"`, `"grpo_clip"`).
- **`run_sft_microbatch_train_step`** / **`run_grpo_microbatch_train_step`** — Full microbatch step including backward pass; must scale by `gradient_accumulation_steps`.

Optional (RLHF supplement):
- **`get_packed_sft_dataset`** — Build packed sequence dataset from JSONL instruction data.
- **`run_iterate_batches`** — DataLoader-style batch iterator over a dataset.
- **`run_parse_mmlu_response`** / **`run_parse_gsm8k_response`** — Response parsers for MMLU (letter choice) and GSM8K (last number).
- **`run_compute_per_instance_dpo_loss`** — DPO loss for a single preference pair.

## Reward Functions

Two reward functions are pre-implemented in `cs336_alignment/drgrpo_grader.py` and used as the GRPO reward signal:

- **`r1_zero_reward_fn`** — Strict format: requires `</think> <answer>...</answer>` structure. Gives `reward=1.0` only if both format and answer are correct.
- **`question_only_reward_fn`** — Only requires `\boxed{}` in the response (no `<think>` tags needed).

Both return `{"reward": float, "format_reward": float, "answer_reward": float}`.

## Snapshot Testing Notes

Pre-computed snapshots live in `tests/_snapshots/` as `.npz` (numpy arrays) and `.pkl` (arbitrary objects). The default tolerance is `rtol=1e-4, atol=1e-2`. Use `--snapshot-exact` for zero tolerance. Snapshots are never auto-updated by student code; they are fixed reference outputs.
