"""Expert Iteration (EI) trainer for MATH.

Algorithm:
  1. Start with initial policy pi_theta
  2. For each EI step:
     a. Sample a batch of questions D_b from D
     b. Generate G outputs per question using vLLM
     c. Score outputs with reward function, keep only correct ones
     d. SFT on the filtered (question, correct_response) pairs
  3. Repeat
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    get_batches,
    get_response_log_probs,
    init_vllm,
    load_data,
    load_policy_into_vllm_instance,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

logger = logging.getLogger(__name__)


@dataclass
class EITrainer:
    """Expert Iteration trainer for MATH reasoning."""

    model_path: str = "Qwen/Qwen2.5-Math-1.5B"
    train_data_path: str = "data/sft-reason/sft_gpt-oss-120b.jsonl"
    val_data_path: str = "data/sft-reason/val.jsonl"
    prompt_template_path: str = "data/sft-reason/r1_zero.prompt"
    output_dir: str = "outputs/ei"

    # EI hyperparameters
    n_ei_steps: int = 3
    questions_per_step: int = 256  # |D_b|
    num_generations: int = 8       # G: number of outputs per question

    # Sampling
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 1024
    sampling_min_tokens: int = 4
    seed: int = 42

    # SFT hyperparameters (for inner SFT loop)
    lr: float = 1e-5
    weight_decay: float = 0.01
    sft_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_grad_norm: float = 1.0
    device: str = "cuda:0"
    flash_attn: bool = True

    # Logging and evaluation
    log_every: int = 10
    max_val_examples: int = 500
    use_wandb: bool = False
    wandb_project: str = "cs336-ei"
    run_name: str | None = None

    # Populated during setup
    policy: PreTrainedModel = field(default=None, init=False, repr=False)
    tokenizer: PreTrainedTokenizerBase = field(default=None, init=False, repr=False)
    prompt_template: str = field(default="", init=False, repr=False)
    train_data: list[dict] = field(default_factory=list, init=False, repr=False)
    val_data: list[dict] | None = field(default=None, init=False, repr=False)

    def setup(self):
        """Load model, tokenizer, and data."""
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

    def _sample_questions(self, rng: random.Random) -> list[dict]:
        """Sample a batch of questions D_b from D."""
        n = min(self.questions_per_step, len(self.train_data))
        return rng.sample(self.train_data, n)

    def _generate_and_filter(
        self,
        questions: list[dict],
        reward_fn: Callable,
        seed: int,
    ) -> tuple[list[str], list[str]]:
        """Generate G outputs per question, score them, and keep only correct ones.

        Returns (prompts, outputs) for SFT training.
        """
        prompts_for_gen = []
        ground_truths = []
        problems = []
        for ex in questions:
            q = ex.get("question", ex.get("problem", ""))
            a = str(ex.get("answer", ex.get("expected_answer", "")))
            prompts_for_gen.append(self.prompt_template.format(question=q))
            ground_truths.append(a)
            problems.append(q)

        sampling_params = SamplingParams(
            temperature=self.sampling_temperature,
            max_tokens=self.sampling_max_tokens,
            min_tokens=self.sampling_min_tokens,
            n=self.num_generations,
            seed=seed,
        )
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True

        # Offload policy to CPU to free VRAM for vLLM
        single_gpu = torch.cuda.device_count() <= 1
        if single_gpu:
            self.policy.cpu()
            torch.cuda.empty_cache()

        gen_device = "cuda:0" if single_gpu else "cuda:1"
        llm = init_vllm(
            model_id=self.model_path,
            device=gen_device,
            seed=seed,
            gpu_memory_utilization=0.85 if single_gpu else 0.4,
        )
        load_policy_into_vllm_instance(self.policy, llm)

        raw_outputs = llm.generate(prompts_for_gen, sampling_params)

        # Filter: keep only correct outputs
        sft_prompts = []
        sft_outputs = []
        total_generated = 0
        total_correct = 0

        for prompt_str, output, gt in zip(prompts_for_gen, raw_outputs, ground_truths):
            for gen in output.outputs:
                total_generated += 1
                generation = gen.text
                scores = reward_fn(generation, gt)
                if scores["reward"] > 0:
                    total_correct += 1
                    sft_prompts.append(prompt_str)
                    sft_outputs.append(generation)

        del llm
        torch.cuda.empty_cache()

        if single_gpu:
            self.policy.to(self.device)

        logger.info(
            f"Generation: {total_generated} total, {total_correct} correct "
            f"({100 * total_correct / max(total_generated, 1):.1f}% pass rate)"
        )
        return sft_prompts, sft_outputs

    def _sft_on_data(
        self,
        prompts: list[str],
        outputs: list[str],
        ei_step: int,
    ) -> dict[str, float]:
        """Run SFT on filtered data (inner loop of EI)."""
        if len(prompts) == 0:
            logger.warning("No correct examples to train on, skipping SFT step.")
            return {"sft_loss": 0.0, "num_examples": 0}

        self.policy.gradient_checkpointing_enable()
        self.policy.train()

        optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        total_steps = (
            (num_batches + self.gradient_accumulation_steps - 1)
            // self.gradient_accumulation_steps
            * self.sft_epochs
        )

        logger.info(
            f"SFT on {len(prompts)} examples, {num_batches} microbatches/epoch, "
            f"~{total_steps} optimizer steps, {self.sft_epochs} epochs"
        )

        global_step = 0
        total_loss = 0.0
        loss_count = 0

        for epoch in range(self.sft_epochs):
            optimizer.zero_grad()
            microbatch_count = 0
            accum_loss = 0.0

            for prompt_batch, output_batch in get_batches(
                prompts, outputs, self.batch_size, shuffle=True, seed=self.seed + ei_step * 100 + epoch
            ):
                batch = tokenize_prompt_and_output(prompt_batch, output_batch, self.tokenizer)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                response_mask = batch["response_mask"].to(self.device)

                result = get_response_log_probs(
                    model=self.policy,
                    input_ids=input_ids,
                    labels=labels,
                )
                log_probs = result["log_probs"]

                num_response_tokens = response_mask.sum().item()
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    normalize_constant=max(num_response_tokens, 1.0),
                )
                accum_loss += loss.item()
                microbatch_count += 1

                if microbatch_count % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    total_loss += accum_loss
                    loss_count += 1

                    if global_step % self.log_every == 0:
                        logger.info(f"  SFT step {global_step}/{total_steps} | loss={accum_loss:.4f}")
                        if self.use_wandb:
                            import wandb
                            wandb.log({"sft/loss": accum_loss, "sft_step": global_step})
                    accum_loss = 0.0

            # Tail step
            if microbatch_count > 0 and microbatch_count % self.gradient_accumulation_steps != 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                total_loss += accum_loss
                loss_count += 1

        self.policy.gradient_checkpointing_disable()

        avg_loss = total_loss / max(loss_count, 1)
        return {"sft_loss": avg_loss, "num_examples": len(prompts), "sft_steps": global_step}

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
            gpu_memory_utilization=0.85 if single_gpu else 0.4,
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
        """Run the Expert Iteration loop."""
        self.setup()

        if reward_fn is None:
            reward_fn = r1_zero_reward_fn

        if self.use_wandb:
            import wandb
            wandb.init(project=self.wandb_project, name=self.run_name, config=self._wandb_config())

        rng = random.Random(self.seed)

        for ei_step in range(1, self.n_ei_steps + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Expert Iteration Step {ei_step}/{self.n_ei_steps}")
            logger.info(f"{'='*60}")

            # Step 3: Sample a batch of questions
            questions = self._sample_questions(rng)
            logger.info(f"Sampled {len(questions)} questions")

            # Steps 5-7: Generate, score, and filter
            sft_prompts, sft_outputs = self._generate_and_filter(
                questions, reward_fn, seed=self.seed + ei_step
            )

            # Step 8: SFT on filtered data
            sft_metrics = self._sft_on_data(sft_prompts, sft_outputs, ei_step)
            logger.info(f"SFT metrics: {sft_metrics}")

            # Evaluate
            if self.val_data:
                self.policy.eval()
                eval_metrics = self._evaluate(reward_fn)
                logger.info(f"Eval after EI step {ei_step}: {eval_metrics}")

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        **eval_metrics,
                        "ei_step": ei_step,
                        "sft/num_correct_examples": sft_metrics["num_examples"],
                        "sft/avg_loss": sft_metrics["sft_loss"],
                    })

            # Save checkpoint
            ckpt_path = Path(self.output_dir) / f"ei-step-{ei_step}"
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
