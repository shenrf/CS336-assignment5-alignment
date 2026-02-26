"""GRPO (Group Relative Policy Optimization) trainer for MATH.

Algorithm (Algorithm 3 from the assignment):
  1. Start with initial policy pi_theta
  2. For each GRPO step:
     a. Sample questions from training data
     b. Generate group_size rollouts per question using vLLM
     c. Score with reward function, compute group-normalized advantages
     d. (If grpo_clip) Compute old log-probs under current policy
     e. Train on rollout batch with policy gradient loss
  3. Repeat
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
    init_vllm,
    load_data,
    load_policy_into_vllm_instance,
    masked_mean,
    tokenize_prompt_and_output,
)

logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainer:
    """GRPO trainer for MATH reasoning."""

    model_path: str = "Qwen/Qwen2.5-Math-1.5B"
    train_data_path: str = "data/sft-reason/sft_gpt-oss-120b.jsonl"
    val_data_path: str = "data/sft-reason/val.jsonl"
    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    output_dir: str = "outputs/grpo"

    # GRPO hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128
    gpu_memory_utilization: float = 0.85
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    cliprange: float = 0.2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)

    # Evaluation and logging
    eval_every: int = 5
    max_val_examples: int = 1024
    seed: int = 42
    device: str = "cuda:0"
    flash_attn: bool = True
    use_wandb: bool = False
    wandb_project: str = "cs336-grpo"
    run_name: str | None = None

    # Populated during setup
    policy: PreTrainedModel = field(default=None, init=False, repr=False)
    tokenizer: PreTrainedTokenizerBase = field(default=None, init=False, repr=False)
    prompt_template: str = field(default="", init=False, repr=False)
    train_data: list[dict] = field(default_factory=list, init=False, repr=False)
    val_data: list[dict] | None = field(default=None, init=False, repr=False)

    @property
    def n_prompts_per_rollout_batch(self) -> int:
        return self.rollout_batch_size // self.group_size

    @property
    def microbatch_size(self) -> int:
        return self.train_batch_size // self.gradient_accumulation_steps

    @property
    def n_microbatches_per_rollout_batch(self) -> int:
        return self.rollout_batch_size // self.microbatch_size

    @property
    def is_off_policy(self) -> bool:
        return self.epochs_per_rollout_batch > 1

    def _sanity_checks(self):
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, (
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
        assert self.rollout_batch_size % self.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        assert self.train_batch_size >= self.group_size, (
            "train_batch_size must be greater than or equal to group_size"
        )
        if self.is_off_policy and self.loss_type != "grpo_clip":
            raise ValueError(
                f"Off-policy training (epochs_per_rollout_batch={self.epochs_per_rollout_batch}) "
                f"requires loss_type='grpo_clip' for importance weighting, "
                f"but got '{self.loss_type}'."
            )

    def setup(self):
        """Load model, tokenizer, and data."""
        self._sanity_checks()
        logging.basicConfig(
            format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.train_data = load_data(self.train_data_path)
        self.prompt_template = Path(self.prompt_template_path).read_text()

        if self.val_data_path:
            self.val_data = load_data(self.val_data_path)

        logger.info(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.policy = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if self.flash_attn else None,
            device_map=self.device,
        )

    def _generate_rollouts(
        self,
        questions: list[dict],
        reward_fn: Callable,
        seed: int,
    ) -> tuple[list[str], list[str], list[str]]:
        """Generate group_size rollouts per question using vLLM.

        Returns (prompt_strs_repeated, response_strs, ground_truths_repeated).
        """
        prompts_for_gen = []
        ground_truths = []
        for ex in questions:
            q = ex.get("question", ex.get("problem", ""))
            a = str(ex.get("answer", ex.get("expected_answer", "")))
            prompts_for_gen.append(self.prompt_template.format(question=q))
            ground_truths.append(a)

        sampling_params = SamplingParams(
            temperature=self.sampling_temperature,
            max_tokens=self.sampling_max_tokens,
            min_tokens=self.sampling_min_tokens,
            n=self.group_size,
            seed=seed,
        )
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True

        # Offload policy to CPU to free VRAM for vLLM on single GPU
        single_gpu = torch.cuda.device_count() <= 1
        if single_gpu:
            self.policy.cpu()
            torch.cuda.empty_cache()

        gen_device = "cuda:0" if single_gpu else "cuda:1"
        llm = init_vllm(
            model_id=self.model_path,
            device=gen_device,
            seed=seed,
            gpu_memory_utilization=self.gpu_memory_utilization if single_gpu else 0.4,
        )
        load_policy_into_vllm_instance(self.policy, llm)

        raw_outputs = llm.generate(prompts_for_gen, sampling_params)

        # Flatten: each question produces group_size responses
        prompt_strs_repeated = []
        response_strs = []
        ground_truths_repeated = []

        for prompt_str, output, gt in zip(prompts_for_gen, raw_outputs, ground_truths):
            for gen in output.outputs:
                prompt_strs_repeated.append(prompt_str)
                response_strs.append(gen.text)
                ground_truths_repeated.append(gt)

        del llm
        torch.cuda.empty_cache()

        if single_gpu:
            self.policy.to(self.device)

        logger.info(f"Generated {len(response_strs)} rollouts from {len(questions)} questions")
        return prompt_strs_repeated, response_strs, ground_truths_repeated

    def _get_old_log_probs(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> list[torch.Tensor]:
        """Compute log-probs under the current (old) policy for grpo_clip.

        Returns a list of detached log_prob tensors, one per microbatch.
        """
        old_log_probs_list = []
        n = len(prompts)
        self.policy.eval()

        with torch.inference_mode():
            for start in range(0, n, self.microbatch_size):
                end = min(start + self.microbatch_size, n)
                batch = tokenize_prompt_and_output(
                    prompts[start:end], responses[start:end], self.tokenizer
                )
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                result = get_response_log_probs(
                    model=self.policy,
                    input_ids=input_ids,
                    labels=labels,
                )
                old_log_probs_list.append(result["log_probs"].detach().cpu())

        return old_log_probs_list

    def _train_on_rollout_batch(
        self,
        prompts: list[str],
        responses: list[str],
        advantages: torch.Tensor,
        raw_rewards: torch.Tensor,
        old_log_probs_list: list[torch.Tensor] | None,
        optimizer: torch.optim.Optimizer,
        grpo_step: int,
    ) -> dict[str, float]:
        """Inner training loop over a rollout batch (Algorithm 3, lines 8-10)."""
        self.policy.gradient_checkpointing_enable()
        self.policy.train()

        n = len(prompts)
        total_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        loss_count = 0
        microbatch_idx = 0

        for epoch in range(self.epochs_per_rollout_batch):
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_entropy = 0.0
            accum_clip = 0.0
            microbatch_in_step = 0

            old_lp_idx = 0
            for start in range(0, n, self.microbatch_size):
                end = min(start + self.microbatch_size, n)
                mb_prompts = prompts[start:end]
                mb_responses = responses[start:end]
                mb_advantages = advantages[start:end]
                mb_raw_rewards = raw_rewards[start:end]

                batch = tokenize_prompt_and_output(
                    mb_prompts, mb_responses, self.tokenizer
                )
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                response_mask = batch["response_mask"].to(self.device)

                result = get_response_log_probs(
                    model=self.policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True,
                )
                policy_log_probs = result["log_probs"]
                token_entropy = result["token_entropy"]

                # Prepare advantages/rewards for this microbatch
                adv = mb_advantages.unsqueeze(-1).to(self.device)
                rr = mb_raw_rewards.unsqueeze(-1).to(self.device)

                # Old log probs for grpo_clip
                old_lp = None
                if self.loss_type == "grpo_clip" and old_log_probs_list is not None:
                    old_lp = old_log_probs_list[old_lp_idx].to(self.device)
                    old_lp_idx += 1

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    loss_type=self.loss_type,
                    raw_rewards=rr,
                    advantages=adv,
                    old_log_probs=old_lp,
                    cliprange=self.cliprange,
                )

                # Track entropy
                mb_entropy = masked_mean(token_entropy, response_mask).item()
                accum_loss += loss.item()
                accum_entropy += mb_entropy
                accum_clip += metadata.get("clip_rate", 0.0)
                microbatch_in_step += 1
                microbatch_idx += 1

                if microbatch_in_step % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.max_grad_norm
                        ).item()
                    else:
                        grad_norm = 0.0
                    optimizer.step()
                    optimizer.zero_grad()

                    avg_loss = accum_loss
                    avg_entropy = accum_entropy / microbatch_in_step
                    avg_clip = accum_clip / microbatch_in_step

                    total_loss += avg_loss
                    total_entropy += avg_entropy
                    total_clip_fraction += avg_clip
                    loss_count += 1

                    logger.info(
                        f"  Step {grpo_step} | opt_step {loss_count} | "
                        f"loss={avg_loss:.4f} grad_norm={grad_norm:.4f} "
                        f"entropy={avg_entropy:.4f} clip_frac={avg_clip:.4f}"
                    )

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/entropy": avg_entropy,
                            "train/clip_fraction": avg_clip,
                            "grpo_step": grpo_step,
                        })

                    accum_loss = 0.0
                    accum_entropy = 0.0
                    accum_clip = 0.0
                    microbatch_in_step = 0

            # Tail step: handle remaining microbatches
            if microbatch_in_step > 0 and microbatch_in_step % self.gradient_accumulation_steps != 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accum_loss
                total_entropy += accum_entropy / microbatch_in_step
                total_clip_fraction += accum_clip / microbatch_in_step
                loss_count += 1

        self.policy.gradient_checkpointing_disable()

        return {
            "train/loss": total_loss / max(loss_count, 1),
            "train/entropy": total_entropy / max(loss_count, 1),
            "train/clip_fraction": total_clip_fraction / max(loss_count, 1),
            "train/avg_reward": raw_rewards.mean().item(),
        }

    def _evaluate(self, reward_fn: Callable) -> dict[str, float]:
        """Evaluate current policy on val set."""
        val_subset = self.val_data[: self.max_val_examples]

        eval_prompts = []
        ground_truths = []
        for ex in val_subset:
            q = ex.get("question", ex.get("problem", ""))
            a = str(ex.get("answer", ex.get("expected_answer", "")))
            eval_prompts.append(self.prompt_template.format(question=q))
            ground_truths.append(a)

        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=self.sampling_max_tokens,
            min_tokens=self.sampling_min_tokens,
        )
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True

        single_gpu = torch.cuda.device_count() <= 1
        if single_gpu:
            self.policy.cpu()
            torch.cuda.empty_cache()

        eval_device = "cuda:0" if single_gpu else "cuda:1"
        llm = init_vllm(
            model_id=self.model_path,
            device=eval_device,
            seed=42,
            gpu_memory_utilization=self.gpu_memory_utilization if single_gpu else 0.4,
        )
        load_policy_into_vllm_instance(self.policy, llm)

        raw_outputs = llm.generate(eval_prompts, sampling_params)

        results = []
        for output, gt in zip(raw_outputs, ground_truths):
            generation = output.outputs[0].text
            scores = reward_fn(generation, gt)
            results.append(scores)

        del llm
        torch.cuda.empty_cache()

        if single_gpu:
            self.policy.to(self.device)

        avg_reward = mean(r["reward"] for r in results)
        avg_format = mean(r["format_reward"] for r in results)
        avg_answer = mean(r["answer_reward"] for r in results)

        return {
            "eval/reward": avg_reward,
            "eval/format_reward": avg_format,
            "eval/answer_reward": avg_answer,
        }

    def _wandb_config(self) -> dict:
        skip = {"policy", "tokenizer", "prompt_template", "train_data", "val_data"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}

    def train(self, reward_fn: Callable | None = None):
        """Run the GRPO training loop (Algorithm 3)."""
        self.setup()

        if reward_fn is None:
            reward_fn = r1_zero_reward_fn

        if self.use_wandb:
            import wandb
            wandb.init(project=self.wandb_project, name=self.run_name, config=self._wandb_config())

        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        rng = random.Random(self.seed)

        mode = "off-policy" if self.is_off_policy else "on-policy"
        logger.info(
            f"GRPO training: {mode}, loss_type={self.loss_type}, "
            f"epochs_per_rollout_batch={self.epochs_per_rollout_batch}, "
            f"microbatch_size={self.microbatch_size}, "
            f"n_microbatches_per_rollout_batch={self.n_microbatches_per_rollout_batch}"
        )

        for step in range(1, self.n_grpo_steps + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"GRPO Step {step}/{self.n_grpo_steps}")
            logger.info(f"{'='*60}")

            # Sample questions for this rollout batch
            n_questions = self.n_prompts_per_rollout_batch
            questions = rng.sample(
                self.train_data, min(n_questions, len(self.train_data))
            )

            # Generate rollouts
            prompts, responses, ground_truths = self._generate_rollouts(
                questions, reward_fn, seed=self.seed + step
            )

            # Compute rewards and group-normalized advantages
            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                reward_fn=reward_fn,
                rollout_responses=responses,
                repeated_ground_truths=ground_truths,
                group_size=self.group_size,
                advantage_eps=self.advantage_eps,
                normalize_by_std=self.use_std_normalization,
            )

            logger.info(
                f"Rewards: mean={raw_rewards.mean():.4f} max={raw_rewards.max():.4f} "
                f"fraction_correct={(raw_rewards > 0).float().mean():.4f}"
            )

            # Compute old log-probs for off-policy / grpo_clip
            old_log_probs_list = None
            if self.loss_type == "grpo_clip":
                logger.info("Computing old log-probs for grpo_clip (off-policy)")
                old_log_probs_list = self._get_old_log_probs(prompts, responses)

            # Train on rollout batch
            train_metrics = self._train_on_rollout_batch(
                prompts=prompts,
                responses=responses,
                advantages=advantages,
                raw_rewards=raw_rewards,
                old_log_probs_list=old_log_probs_list,
                optimizer=optimizer,
                grpo_step=step,
            )
            logger.info(f"Train metrics: {train_metrics}")

            # Evaluate
            if self.val_data and step % self.eval_every == 0:
                self.policy.eval()
                eval_metrics = self._evaluate(reward_fn)
                logger.info(f"Eval at step {step}: {eval_metrics}")

                if self.use_wandb:
                    import wandb
                    wandb.log({**eval_metrics, "grpo_step": step})

            # Save checkpoint
            ckpt_path = Path(self.output_dir) / f"grpo-step-{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self.policy.save_pretrained(ckpt_path)
            self.tokenizer.save_pretrained(ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # Save final model
        final_path = Path(self.output_dir) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Saved final model to {final_path}")

        if self.use_wandb:
            import wandb
            wandb.finish()


def main():
    import typer

    app = typer.Typer(pretty_exceptions_enable=False)

    @app.command()
    def train(
        model_path: str = "Qwen/Qwen2.5-Math-1.5B",
        train_data_path: str = "data/sft-reason/sft_gpt-oss-120b.jsonl",
        val_data_path: str = "data/sft-reason/val.jsonl",
        prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt",
        output_dir: str = "outputs/grpo",
        n_grpo_steps: int = 200,
        learning_rate: float = 1e-5,
        advantage_eps: float = 1e-6,
        rollout_batch_size: int = 256,
        group_size: int = 8,
        sampling_temperature: float = 1.0,
        sampling_min_tokens: int = 4,
        sampling_max_tokens: int = 1024,
        epochs_per_rollout_batch: int = 1,
        train_batch_size: int = 256,
        gradient_accumulation_steps: int = 128,
        gpu_memory_utilization: float = 0.85,
        loss_type: str = "reinforce_with_baseline",
        use_std_normalization: bool = True,
        cliprange: float = 0.2,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.0,
        eval_every: int = 5,
        max_val_examples: int = 1024,
        seed: int = 42,
        flash_attn: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "cs336-grpo",
        run_name: str | None = None,
    ):
        trainer = GRPOTrainer(
            model_path=model_path,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            prompt_template_path=prompt_template_path,
            output_dir=output_dir,
            n_grpo_steps=n_grpo_steps,
            learning_rate=learning_rate,
            advantage_eps=advantage_eps,
            rollout_batch_size=rollout_batch_size,
            group_size=group_size,
            sampling_temperature=sampling_temperature,
            sampling_min_tokens=sampling_min_tokens,
            sampling_max_tokens=sampling_max_tokens,
            epochs_per_rollout_batch=epochs_per_rollout_batch,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gpu_memory_utilization=gpu_memory_utilization,
            loss_type=loss_type,
            use_std_normalization=use_std_normalization,
            cliprange=cliprange,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            eval_every=eval_every,
            max_val_examples=max_val_examples,
            seed=seed,
            flash_attn=flash_attn,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            run_name=run_name,
        )
        trainer.train()

    app()


if __name__ == "__main__":
    main()
