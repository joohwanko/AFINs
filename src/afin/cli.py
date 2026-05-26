"""CLI entry point for AFIN training."""
import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch

from .eval import evaluate_model, make_eval_specs, prepare_eval_suite, select_stratified_specs, summarize_records
from .tasks import (
    LIKELIHOOD_FAMILIES,
    PRIOR_FAMILIES,
    canonicalize_posterior_family,
    make_exp_type,
    parse_family_list,
)
from .train import (
    build_model_kwargs,
    cleanup_runtime,
    distributed_barrier,
    init_wandb,
    make_run_dir,
    save_checkpoint,
    setup_runtime,
    train,
    write_json,
)


def _default_workers():
    return max(1, min(4, os.cpu_count() or 1))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--posterior-family", type=canonicalize_posterior_family, default="gaussian",
                   choices=["gaussian", "flow"])
    p.add_argument("--prior-families",
                   type=lambda v: parse_family_list(v, allowed=set(PRIOR_FAMILIES), name="prior"),
                   default=list(PRIOR_FAMILIES))
    p.add_argument("--likelihood-families",
                   type=lambda v: parse_family_list(v, allowed=set(LIKELIHOOD_FAMILIES), name="likelihood"),
                   default=list(LIKELIHOOD_FAMILIES))

    p.add_argument("--d-min", type=int, default=1)
    p.add_argument("--d-max", type=int, default=16)
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=256)

    # Model
    p.add_argument("--feat-dim", type=int, default=40)
    p.add_argument("--hidden-dim", type=int, default=160)
    p.add_argument("--adapter-hidden-dim", type=int, default=64)
    p.add_argument("--adapter-num-layers", type=int, default=2)
    p.add_argument("--box-hidden-dim", type=int, default=192)
    p.add_argument("--box-num-layers", type=int, default=3)
    p.add_argument("--box-depth", type=int, default=4)
    p.add_argument("--merge-model-dim", type=int, default=192)
    p.add_argument("--merge-num-layers", type=int, default=4)
    p.add_argument("--decoder-hidden-dim", type=int, default=None)
    p.add_argument("--decoder-num-layers", type=int, default=3)
    p.add_argument("--decoder-residual-scale", type=float, default=0.1)
    p.add_argument("--flow-hidden-dim", type=int, default=32)
    p.add_argument("--flow-context-dim", type=int, default=None)
    p.add_argument("--flow-num-layers", type=int, default=3)
    p.add_argument("--flow-num-steps", type=int, default=4)
    p.add_argument("--flow-max-log-scale", type=float, default=1.2)
    p.add_argument("--flow-start-step", type=int, default=0)
    p.add_argument("--flow-warmup-steps", type=int, default=2500)
    p.add_argument("--flow-max-strength", type=float, default=1.0)

    # Training
    p.add_argument("--num-steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--task-microbatches", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16"])
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--encoder-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--merge-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=0)

    # Sampling biases
    p.add_argument("--hard-bias-prob", type=float, default=0.6)
    p.add_argument("--hard-d-alpha", type=float, default=1.0)
    p.add_argument("--hard-n-alpha", type=float, default=0.75)
    p.add_argument("--train-x-kappa-max", type=float, default=40.0)

    # Eval / NUTS reference
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--eval-num-samples", type=int, default=512)
    p.add_argument("--eval-sample-chunk-size", type=int, default=64)
    p.add_argument("--ref-num-samples", type=int, default=512)
    p.add_argument("--ref-num-warmup", type=int, default=1000)
    p.add_argument("--ref-source-num-samples", type=int, default=2048)
    p.add_argument("--ref-source-num-warmup", type=int, default=None)
    p.add_argument("--eval-cache-workers", type=int, default=_default_workers())
    p.add_argument("--eval-cache-dir", type=str, default="eval_cache")
    p.add_argument("--checkpoint-root", type=str, default="checkpoints")

    # Wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="afin")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def main():
    args = parse_args()
    runtime = setup_runtime()
    seed = int(args.seed) + int(runtime["rank"] if runtime["use_ddp"] else 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = runtime["device"]
    is_main = runtime["is_main_process"]
    wandb_run = None

    try:
        exp_type = make_exp_type(
            d_min=args.d_min, d_max=args.d_max, n_min=args.n_min, n_max=args.n_max,
            posterior_family=args.posterior_family,
            prior_families=args.prior_families, likelihood_families=args.likelihood_families,
        )

        run_dir = None
        eval_suite = None

        if is_main:
            run_dir = make_run_dir(exp_type=exp_type, checkpoint_root=args.checkpoint_root)
            write_json(
                Path(run_dir) / "config.json",
                {"args": vars(args), "exp_type": asdict(exp_type), "device": device, "seed": seed},
            )
            print(f"Checkpoint dir: {run_dir}")
            wandb_run = init_wandb(args=args, exp_type=exp_type, run_dir=run_dir)
            online_specs = select_stratified_specs(make_eval_specs(exp_type=exp_type), exp_type=exp_type)
            eval_suite = prepare_eval_suite(
                exp_type=exp_type,
                ref_num_samples=args.ref_num_samples,
                ref_num_warmup=args.ref_num_warmup,
                ref_source_num_samples=args.ref_source_num_samples,
                ref_source_num_warmup=args.ref_source_num_warmup,
                cache_dir=args.eval_cache_dir,
                seed=args.seed,
                specs=online_specs,
                workers=args.eval_cache_workers,
            )

        if runtime["use_ddp"]:
            distributed_barrier()

        results = train(
            exp_type=exp_type,
            model_kwargs=build_model_kwargs(args),
            amp_mode=args.amp,
            lr=args.lr,
            wd=args.wd,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            log_every=args.log_every,
            device=device,
            eval_suite=eval_suite,
            eval_every=args.eval_every,
            eval_num_samples=args.eval_num_samples,
            eval_sample_chunk_size=args.eval_sample_chunk_size,
            task_microbatches=args.task_microbatches,
            use_ema=args.ema,
            run_dir=run_dir,
            save_every=args.save_every,
            wandb_run=wandb_run,
            flow_start_step=args.flow_start_step,
            flow_warmup_steps=args.flow_warmup_steps,
            flow_max_strength=args.flow_max_strength,
            hard_bias_prob=args.hard_bias_prob,
            hard_d_alpha=args.hard_d_alpha,
            hard_n_alpha=args.hard_n_alpha,
            train_x_kappa_max=args.train_x_kappa_max,
            use_ddp=runtime["use_ddp"],
            ddp_local_rank=runtime["local_rank"],
        )

        if is_main and run_dir is not None:
            save_checkpoint(
                Path(run_dir) / "final.pt",
                model=results["model"], ema=results["ema"],
                optimizer=None, scheduler=None,
                step=args.num_steps, exp_type=exp_type, metrics=None,
                extra={"best_score": results["best_score"]},
            )
            if eval_suite is not None:
                final_model = results["ema_model"] if results["ema_model"] is not None else results["model"]
                summary = summarize_records(
                    evaluate_model(
                        final_model, eval_suite,
                        num_samples=args.eval_num_samples, sample_chunk_size=args.eval_sample_chunk_size,
                    )
                )
                write_json(Path(run_dir) / "final_eval.json", {"overall": summary["overall"], "by_x_family": summary["by_x_family"]})
                print(
                    f"[final-eval] m1={summary['overall'].get('m1', float('nan')):.4f} "
                    f"m2={summary['overall'].get('m2', float('nan')):.4f} "
                    f"ce/d={summary['overall'].get('cross_entropy_p_to_q_per_dim', float('nan')):.4f}"
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_runtime()


if __name__ == "__main__":
    main()
