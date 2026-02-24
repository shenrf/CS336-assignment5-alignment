import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    format_sft_examples,
    get_batches,
    get_response_log_probs,
    init_vllm,
    load_data,
    load_policy_into_vllm_instance,
    masked_normalize,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility with imports from sft
load_sft_data = load_data


# ---------------------------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------------------------

@dataclass
class SFTTrainer:
    """Supervised fine-tuning trainer for causal language models."""

    model_path: str = "Qwen/Qwen2.5-Math-1.5B"
    train_data_path: str = "data/sft-reason/sft_gpt-oss-120b.jsonl"
    val_data_path: str = "data/sft-reason/val.jsonl"
    prompt_template_path: str = "data/sft-reason/r1_zero.prompt"
    output_dir: str = "outputs/sft_filtered"

    # Training hyperparameters (tuned for 12GB VRAM, e.g. RTX 4070)
    lr: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_grad_norm: float = 1.0
    device: str = "cuda:0"
    flash_attn: bool = True

    # Logging and evaluation
    log_every: int = 10
    eval_every: int = 100
    max_val_examples: int = 500
    save_every: int = 200
    use_wandb: bool = False
    wandb_project: str = "cs336-sft"
    run_name: str | None = None

    # Populated during setup
    policy: PreTrainedModel = field(default=None, init=False, repr=False)
    tokenizer: PreTrainedTokenizerBase = field(default=None, init=False, repr=False)
    optimizer: torch.optim.Optimizer = field(default=None, init=False, repr=False)
    prompt_template: str = field(default="", init=False, repr=False)
    train_prompts: list[str] = field(default_factory=list, init=False, repr=False)
    train_outputs: list[str] = field(default_factory=list, init=False, repr=False)
    val_data: list[dict] | None = field(default=None, init=False, repr=False)

    def setup(self):
        """Load model, tokenizer, data, and create optimizer."""
        logging.basicConfig(
            format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        # Load data
        train_data = load_data(self.train_data_path)
        self.prompt_template = Path(self.prompt_template_path).read_text()
        self.train_prompts, self.train_outputs = format_sft_examples(train_data, self.prompt_template)

        if self.val_data_path:
            self.val_data = load_data(self.val_data_path)

        # Load model and tokenizer
        logger.info(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.policy = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if self.flash_attn else None,
            device_map=self.device,
        )
        self.policy.gradient_checkpointing_enable()
        self.policy.train()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _run_vllm_eval(
        self,
        val_examples: list[dict],
        reward_fn: Callable,
    ) -> list[dict]:
        """Shared vLLM generation + scoring logic.

        Offloads policy to CPU on single-GPU, runs vLLM, scores results,
        then moves policy back. Returns per-example result dicts.
        """
        eval_prompts = []
        ground_truths = []
        for ex in val_examples:
            q = ex.get("question", ex.get("problem", ""))
            a = str(ex.get("answer", ex.get("expected_answer", "")))
            eval_prompts.append(self.prompt_template.format(question=q))
            ground_truths.append(a)

        sampling_params = SamplingParams(temperature=1.0, max_tokens=1024, top_p=1.0)
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True

        # Offload policy to CPU to free VRAM for vLLM on single-GPU
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
        for output, ex, gt in zip(raw_outputs, val_examples, ground_truths):
            generation = output.outputs[0].text
            scores = reward_fn(generation, gt)
            results.append({
                "problem": ex.get("question", ex.get("problem", "")),
                "expected_answer": gt,
                "output": generation,
                "reward": {
                    "reward": scores["reward"],
                    "format_reward": scores["format_reward"],
                    "answer_reward": scores["answer_reward"],
                },
            })

        del llm
        torch.cuda.empty_cache()

        if single_gpu:
            self.policy.to(self.device)

        return results

    def evaluate(self, reward_fn: Callable | None = None) -> dict[str, float]:
        """Quick evaluation on a subset of the val set. Returns summary metrics."""
        if reward_fn is None:
            reward_fn = r1_zero_reward_fn

        val_subset = self.val_data[: self.max_val_examples]
        results = self._run_vllm_eval(val_subset, reward_fn)

        avg_acc = mean(r["reward"]["answer_reward"] for r in results)
        avg_format_acc = mean(r["reward"]["format_reward"] for r in results)
        return {
            "eval/reward": mean(r["reward"]["reward"] for r in results),
            "eval/format_reward": avg_format_acc,
            "eval/answer_reward": avg_acc,
            "accuracy": {
                "avg_acc": round(avg_acc, 4),
                "avg_format_acc": round(avg_format_acc, 4),
            },
        }

    def _wandb_config(self) -> dict:
        """Return serializable config for wandb (exclude model/optimizer objects)."""
        skip = {"policy", "tokenizer", "optimizer", "train_prompts", "train_outputs", "val_data", "prompt_template"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}

    def train(self):
        """Run the full training loop."""
        self.setup()

        if self.use_wandb:
            import wandb
            wandb.init(project=self.wandb_project, name=self.run_name, config=self._wandb_config())
            wandb.define_metric("train_step")
            wandb.define_metric("eval_step")
            wandb.define_metric("train/*", step_metric="train_step")
            wandb.define_metric("eval/*", step_metric="eval_step")

        global_step = 0
        eval_step = 0
        num_batches_per_epoch = (len(self.train_prompts) + self.batch_size - 1) // self.batch_size
        total_steps = (
            (num_batches_per_epoch + self.gradient_accumulation_steps - 1)
            // self.gradient_accumulation_steps
            * self.epochs
        )

        total_microbatches = num_batches_per_epoch * self.epochs
        logger.info(f"Training for {self.epochs} epochs, ~{total_steps} optimizer steps, {total_microbatches} microbatches")
        logger.info(f"Batch size: {self.batch_size}, Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")

        for epoch in range(self.epochs):
            logger.info(f"=== Epoch {epoch + 1}/{self.epochs} ===")
            self.optimizer.zero_grad()
            microbatch_count = 0
            accum_loss = 0.0

            for batch_idx, (prompt_batch, output_batch) in enumerate(
                get_batches(self.train_prompts, self.train_outputs, self.batch_size, shuffle=True, seed=42 + epoch)
            ):
                batch = tokenize_prompt_and_output(prompt_batch, output_batch, self.tokenizer)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                response_mask = batch["response_mask"].to(self.device)

                # Forward pass (model already in bf16, no autocast needed)
                result = get_response_log_probs(
                    model=self.policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                log_probs = result["log_probs"]

                # Loss + backward (normalize by total response tokens in batch)
                num_response_tokens = response_mask.sum().item()
                loss, meta = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    normalize_constant=max(num_response_tokens, 1.0),
                )
                accum_loss += loss.item()

                microbatch_count += 1
                seq_len = input_ids.shape[-1]
                logger.info(
                    f"  microbatch {microbatch_count}/{total_microbatches} "
                    f"(accum {((microbatch_count - 1) % self.gradient_accumulation_steps) + 1}"
                    f"/{self.gradient_accumulation_steps}) | "
                    f"seq_len={seq_len} | loss={loss.item():.4f}"
                    f" | global Step {global_step}"
                    f" | microbatch Step {microbatch_count}"
                )

                if microbatch_count % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.log_every == 0:
                        logger.info(f"Step {global_step}/{total_steps} | Loss: {accum_loss:.4f}")
                        if self.use_wandb:
                            import wandb
                            wandb.log({"train/loss": accum_loss, "train_step": global_step})
                    accum_loss = 0.0

                    # Evaluate
                    if self.eval_every > 0 and self.val_data and global_step % self.eval_every == 0:
                        self.policy.eval()
                        eval_metrics = self.evaluate()
                        self.policy.train()
                        eval_step += 1
                        logger.info(f"Eval step {eval_step}: {eval_metrics}")
                        if self.use_wandb:
                            import wandb
                            wandb.log({**eval_metrics, "eval_step": eval_step})

            # Step on any remaining accumulated gradients at end of epoch
            if microbatch_count > 0 and microbatch_count % self.gradient_accumulation_steps != 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
                logger.info(f"Step {global_step}/{total_steps} (tail) | Loss: {accum_loss:.4f}")
                accum_loss = 0.0

        # Save final model
        final_path = Path(self.output_dir) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Saved final model to {final_path}")

        # Final evaluation on full MATH val set with detailed report
        if self.val_data:
            logger.info("Running final MATH evaluation on full val set...")
            self.policy.eval()
            report = self.evaluate_full()
            final_metrics = {
                "eval/reward": mean(r["reward"]["reward"] for r in report["results"]),
                "eval/format_reward": mean(r["reward"]["format_reward"] for r in report["results"]),
                "eval/answer_reward": mean(r["reward"]["answer_reward"] for r in report["results"]),
            }
            if self.use_wandb:
                import wandb
                wandb.log({**final_metrics, "eval_step": eval_step + 1})

        if self.use_wandb:
            import wandb
            wandb.finish()

    def evaluate_full(self, reward_fn: Callable | None = None) -> dict:
        """Run evaluation on the full val set and produce a detailed report."""
        if reward_fn is None:
            reward_fn = r1_zero_reward_fn

        results = self._run_vllm_eval(self.val_data, reward_fn)

        avg_acc = mean(r["reward"]["answer_reward"] for r in results)
        avg_format_acc = mean(r["reward"]["format_reward"] for r in results)
        avg_reward = mean(r["reward"]["reward"] for r in results)

        output_path = Path(self.output_dir) / "math_eval_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "results": results,
            "accuracy": {
                "avg_acc": round(avg_acc, 4),
                "avg_format_acc": round(avg_format_acc, 4),
            },
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Wrote {len(results)} results to {output_path}")

        correct, wrong_answer, bad_format = [], [], []
        for r in results:
            fmt = r["reward"]["format_reward"]
            ans = r["reward"]["answer_reward"]
            if fmt == 1.0 and ans == 1.0:
                correct.append(r)
            elif fmt == 1.0 and ans == 0.0:
                wrong_answer.append(r)
            else:
                bad_format.append(r)

        total = len(results)
        n_examples = 5
        print("\n=== MATH Evaluation Results (Post-SFT) ===")
        print(f"Total examples : {total}")
        print(f"Correct        (fmt=1, ans=1): {len(correct):5d}  ({100 * len(correct) / total:.1f}%)")
        print(f"Wrong answer   (fmt=1, ans=0): {len(wrong_answer):5d}  ({100 * len(wrong_answer) / total:.1f}%)")
        print(f"Bad format     (fmt=0, ans=0): {len(bad_format):5d}  ({100 * len(bad_format) / total:.1f}%)")
        print(f"Average reward : {avg_reward:.4f}")

        for label, examples in [
            ("Correct (format=1, answer=1)", correct),
            ("Wrong answer (format=1, answer=0)", wrong_answer),
            ("Bad format (format=0, answer=0)", bad_format),
        ]:
            print(f"\n{'='*70}")
            print(f"  {label}  ({len(examples)} total, showing {min(n_examples, len(examples))})")
            print(f"{'='*70}")
            for i, ex in enumerate(examples[:n_examples]):
                print(f"\n--- Example {i+1} ---")
                print(f"Problem:  {ex['problem'][:200]}")
                print(f"Expected: {ex['expected_answer']}")
                print(f"Output:   {ex['output'][:400]}")
                print(f"Rewards:  format={ex['reward']['format_reward']}  answer={ex['reward']['answer_reward']}")

        return data

    # def _save_checkpoint(self, step: int):
    #     save_path = Path(self.output_dir) / f"checkpoint-{step}"
    #     save_path.mkdir(parents=True, exist_ok=True)
    #     self.policy.save_pretrained(save_path)
    #     self.tokenizer.save_pretrained(save_path)
    #     logger.info(f"Saved checkpoint to {save_path}")
