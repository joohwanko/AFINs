"""Training loop, DDP/AMP setup, and checkpointing for AFIN."""
import contextlib
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .eval import (
    aggregate_x_family_stats,
    evaluate_model,
    filter_metrics,
    summarize_records,
    x_family_eval_log,
    x_family_eval_summary,
    x_family_train_stats,
)
from .model import AFIN, EMA
from .tasks import X_FAMILIES, sample_task_batch


TRAIN_LOSS_EMA_DECAY = 0.95


# -----------------------------------------------------------------------------
# DDP / runtime
# -----------------------------------------------------------------------------


def is_cuda_device(device):
    return str(device).startswith("cuda")


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def distributed_rank():
    return dist.get_rank() if is_distributed() else 0


def distributed_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return distributed_rank() == 0


def distributed_barrier():
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def setup_runtime():
    launched_with_torchrun = any(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = bool(launched_with_torchrun and world_size > 1)
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs = {"backend": backend, "init_method": "env://"}
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
            init_kwargs["device_id"] = torch.device(device)
        else:
            device = "cpu"
        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available() and use_ddp:
            device = f"cuda:{local_rank}"
    if is_distributed():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = max(1, world_size)
        rank, local_rank = 0, 0
    return {
        "use_ddp": use_ddp,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
        "is_main_process": rank == 0,
    }


def cleanup_runtime():
    if is_distributed():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# AMP / LR / model kwargs / IO helpers
# -----------------------------------------------------------------------------


def resolve_amp_dtype(amp_mode, device):
    amp_mode = amp_mode.lower()
    if amp_mode == "off" or not is_cuda_device(device):
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_mode]


def build_lr_scheduler(optimizer, lr, num_steps, warmup_steps):
    total_steps = max(1, int(num_steps))
    warmup_steps = max(0, int(warmup_steps))
    main_steps = max(1, total_steps - warmup_steps)
    main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=lr * 0.1)
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[warmup_steps]
        )
    return main


def build_model_kwargs(args):
    hidden = args.hidden_dim
    return {
        "feat_dim": args.feat_dim,
        "adapter_hidden_dim": args.adapter_hidden_dim,
        "adapter_num_layers": args.adapter_num_layers,
        "box_hidden_dim": args.box_hidden_dim or hidden,
        "box_num_layers": args.box_num_layers,
        "box_depth": args.box_depth,
        "merge_model_dim": args.merge_model_dim or hidden,
        "merge_num_layers": args.merge_num_layers,
        "decoder_hidden_dim": args.decoder_hidden_dim or hidden,
        "decoder_num_layers": args.decoder_num_layers,
        "decoder_residual_scale": args.decoder_residual_scale,
        "flow_hidden_dim": args.flow_hidden_dim or hidden,
        "flow_context_dim": args.flow_context_dim or hidden,
        "flow_num_layers": args.flow_num_layers,
        "flow_num_steps": args.flow_num_steps,
        "flow_max_log_scale": args.flow_max_log_scale,
        "use_encoder_checkpoint": args.encoder_checkpoint,
        "use_merge_checkpoint": args.merge_checkpoint,
    }


def make_run_dir(exp_type, checkpoint_root):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"d_{exp_type.d_min}_{exp_type.d_max}_N_{exp_type.N_min}_{exp_type.N_max}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"d_{exp_type.d_min}_{exp_type.d_max}_N_{exp_type.N_min}_{exp_type.N_max}_{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def update_ema_scalar(current, value):
    value = float(value)
    if current is None:
        return value
    return TRAIN_LOSS_EMA_DECAY * float(current) + (1.0 - TRAIN_LOSS_EMA_DECAY) * value


# -----------------------------------------------------------------------------
# Checkpoint / freeze
# -----------------------------------------------------------------------------


def _unwrap(model):
    while isinstance(model, DDP):
        model = model.module
    return model


def save_checkpoint(path, model, ema, optimizer, scheduler, step, exp_type, metrics=None, extra=None):
    payload = {
        "step": step,
        "exp_type": asdict(exp_type),
        "model_state_dict": _unwrap(model).state_dict(),
        "ema_model_state_dict": None if ema is None else ema.module.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "metrics": metrics,
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def freeze_module(model, module_name):
    module = getattr(model, module_name, None)
    frozen = 0
    if module is None:
        return frozen
    for param in module.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            frozen += param.numel()
    return frozen


def freeze_inactive_flow_for_gaussian(model):
    """Gaussian posterior doesn't use flow modules — freeze them."""
    return sum(freeze_module(model, name) for name in ("flow_context_proj", "flow_affine", "flow"))


# -----------------------------------------------------------------------------
# Wandb
# -----------------------------------------------------------------------------


def init_wandb(args, exp_type, run_dir):
    if not args.wandb:
        return None
    os.environ.setdefault("WANDB_DIR", str(run_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(Path(run_dir) / ".wandb_cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(Path(run_dir) / ".wandb_config"))
    os.environ.setdefault("WANDB_DATA_DIR", str(Path(run_dir) / ".wandb_data"))
    import wandb

    config = {
        "exp_type": asdict(exp_type),
        "model_kwargs": build_model_kwargs(args),
        "checkpoint_dir": str(run_dir),
        "task_microbatches": args.task_microbatches,
        "hard_bias_prob": args.hard_bias_prob,
        "hard_d_alpha": args.hard_d_alpha,
        "hard_n_alpha": args.hard_n_alpha,
    }
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_name or Path(run_dir).name, mode=args.wandb_mode,
        dir=str(run_dir), config=config,
    )


# -----------------------------------------------------------------------------
# Hard-biased (d, N) sampler
# -----------------------------------------------------------------------------


def build_hard_dn_distribution(exp_type, d_alpha=1.0, n_alpha=0.5):
    pairs, weights = [], []
    d_span = max(1, exp_type.d_max - exp_type.d_min + 1)
    n_min = max(1, exp_type.N_min)
    for d in range(exp_type.d_min, exp_type.d_max + 1):
        d_weight = ((d - exp_type.d_min + 1) / d_span) ** float(d_alpha)
        for N in range(exp_type.N_min, exp_type.N_max + 1):
            n_weight = (float(n_min) / float(N)) ** float(n_alpha)
            pairs.append((d, N))
            weights.append(d_weight * n_weight)
    probs = torch.tensor(weights, dtype=torch.double)
    probs = probs / probs.sum().clamp_min(1e-30)
    return pairs, probs


# -----------------------------------------------------------------------------
# Training objective
# -----------------------------------------------------------------------------


class Objective(nn.Module):
    def __init__(self, posterior):
        super().__init__()
        self.posterior = posterior

    def forward(self, task):
        return -self.posterior.log_prob(task, task.z0) / task.d


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(
    *,
    exp_type,
    model_kwargs,
    amp_mode,
    lr,
    wd,
    num_steps,
    warmup_steps,
    batch_size,
    log_every,
    device,
    eval_suite,
    eval_every,
    eval_num_samples,
    eval_sample_chunk_size,
    task_microbatches,
    use_ema,
    run_dir,
    save_every,
    wandb_run,
    flow_start_step,
    flow_warmup_steps,
    flow_max_strength,
    hard_bias_prob,
    hard_d_alpha,
    hard_n_alpha,
    train_x_kappa_max,
    use_ddp,
    ddp_local_rank,
):
    amp_dtype = resolve_amp_dtype(amp_mode, device)
    is_main = is_main_process()
    model_kwargs = dict(model_kwargs)
    model_kwargs["amp_dtype"] = amp_dtype
    model = AFIN(exp_type=exp_type, **model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(
            f"#params={num_params:,} batch={batch_size} amp={amp_mode} "
            f"posterior={exp_type.posterior_family} ddp={use_ddp} world={distributed_world_size()}"
        )

    ema = EMA(model, decay=0.999, update_after_step=200) if use_ema else None
    if is_main:
        print(f"[ema] {'enabled' if ema is not None else 'disabled'}")

    if exp_type.posterior_family == "flow":
        frozen = freeze_module(model, "decoder")
        if ema is not None:
            freeze_module(ema.module, "decoder")
        if is_main and frozen:
            print(f"[freeze] decoder frozen for direct flow training ({frozen:,} params)")
    else:
        frozen = freeze_inactive_flow_for_gaussian(model)
        if ema is not None:
            freeze_inactive_flow_for_gaussian(ema.module)
        if is_main and frozen:
            print(f"[freeze] inactive flow modules frozen ({frozen:,} params)")

    train_model = Objective(model)
    if use_ddp:
        ddp_kwargs = {"find_unused_parameters": True}
        if is_cuda_device(device):
            ddp_kwargs["device_ids"] = [int(ddp_local_rank)]
            ddp_kwargs["output_device"] = int(ddp_local_rank)
        train_model = DDP(train_model, **ddp_kwargs)

    trainable = [p for p in train_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters remain after freeze configuration.")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_mode.lower() == "fp16" and is_cuda_device(device)))
    scheduler = build_lr_scheduler(optimizer, lr, num_steps, warmup_steps)

    eval_enabled = eval_suite is not None
    eval_every = eval_every or log_every
    save_every = save_every or eval_every

    dn_pairs, hard_dn_probs = build_hard_dn_distribution(
        exp_type, d_alpha=hard_d_alpha, n_alpha=hard_n_alpha
    )

    def sample_d_n():
        if hard_bias_prob <= 0.0 or (hard_bias_prob < 1.0 and bool((torch.rand(()) >= hard_bias_prob).item())):
            return (
                int(torch.randint(exp_type.d_min, exp_type.d_max + 1, (1,)).item()),
                int(torch.randint(exp_type.N_min, exp_type.N_max + 1, (1,)).item()),
            )
        idx = int(torch.multinomial(hard_dn_probs, 1).item())
        return dn_pairs[idx]

    def next_specs(num):
        return [
            {"d": d, "N": N, "x_mode": "whitened_like", "x_kappa_max": train_x_kappa_max}
            for d, N in (sample_d_n() for _ in range(max(1, int(num))))
        ]

    loss_ema = None
    best_score = float("inf")
    skip_total = skip_consecutive = 0

    train_model.train()
    for step in range(num_steps):
        if exp_type.posterior_family == "flow":
            if step < flow_start_step:
                flow_strength = 0.0
            else:
                flow_strength = min(
                    float(flow_max_strength),
                    float(flow_max_strength) * float(step - flow_start_step + 1) / float(max(1, flow_warmup_steps)),
                )
        else:
            flow_strength = 1.0
        model.set_flow_strength(flow_strength)
        optimizer.zero_grad(set_to_none=True)

        micro_loss = []
        micro_summaries = []
        specs = next_specs(task_microbatches)
        try:
            for micro_idx, spec in enumerate(specs):
                batch_task = sample_task_batch(batch_size=batch_size, exp_type=exp_type, device=device, spec=spec)
                sync_ctx = (
                    train_model.no_sync()
                    if use_ddp and micro_idx + 1 < len(specs)
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    raw_loss_values = train_model(batch_task)
                    raw_loss = raw_loss_values.mean()
                    scaled = raw_loss / float(len(specs))
                    if scaler.is_enabled():
                        scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                micro_loss.append(raw_loss.item())
                micro_summaries.append(
                    {
                        "d": batch_task.d,
                        "N": batch_task.N,
                        "prior_family": batch_task.prior_family,
                        "likelihood_family": batch_task.likelihood_family,
                        "likelihood_hetero_frac": float(
                            batch_task.meta["likelihood_is_heterogeneous"].float().mean().item()
                        ),
                        "x_condition_number_mean": float(batch_task.meta["x_condition_number"].float().mean().item()),
                        "raw_loss": float(raw_loss.item()),
                        **x_family_train_stats(
                            batch_task.meta["x_family_id"].detach().cpu(),
                            raw_loss_values.detach().cpu(),
                        ),
                    }
                )
        except torch._C._LinAlgError as exc:
            optimizer.zero_grad(set_to_none=True)
            skip_total += 1
            skip_consecutive += 1
            if is_main:
                print(f"[train:skip] step={step} skipped (total={skip_total}, consecutive={skip_consecutive}): {exc}")
            if skip_consecutive >= 12 or (step >= 5000 and skip_total >= 25 and skip_total / float(step + 1) >= 0.01):
                raise RuntimeError(f"Too many SPD failures (total={skip_total})") from exc
            continue
        skip_consecutive = 0

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model, step)

        loss_item = sum(micro_loss) / max(1, len(micro_loss))
        loss_ema = update_ema_scalar(loss_ema, loss_item)

        if is_main and wandb_run is not None:
            agg = aggregate_x_family_stats(micro_summaries)
            log = {
                "train/loss": loss_item,
                "train/loss_ema": loss_ema,
                "train/lr": scheduler.get_last_lr()[0],
                "train/flow_strength": flow_strength,
                "train/d": sum(s["d"] for s in micro_summaries) / len(micro_summaries),
                "train/N": sum(s["N"] for s in micro_summaries) / len(micro_summaries),
                "train/x_condition_number_mean": sum(s["x_condition_number_mean"] for s in micro_summaries) / len(micro_summaries),
                "train/likelihood_hetero_frac": sum(s["likelihood_hetero_frac"] for s in micro_summaries) / len(micro_summaries),
            }
            for family_name in X_FAMILIES:
                log[f"train/x_family_frac_{family_name}"] = agg[f"x_family_frac_{family_name}"]
                if math.isfinite(agg[f"x_family_raw_loss_{family_name}"]):
                    log[f"train/x_family_raw_loss_{family_name}"] = agg[f"x_family_raw_loss_{family_name}"]
            wandb_run.log(log, step=step)

        if is_main and (step % log_every == 0 or step == num_steps - 1):
            last = micro_summaries[-1]
            print(
                f"step={step} loss={loss_item:.4f} ema={loss_ema:.4f} "
                f"d={last['d']} N={last['N']} prior={last['prior_family']} like={last['likelihood_family']}"
            )

        # ---- Eval
        metrics = None
        eval_due = bool(eval_enabled) and (step % eval_every == 0 or step == num_steps - 1)
        if use_ddp and eval_due:
            distributed_barrier()
        if is_main and eval_suite is not None and eval_due:
            eval_model = ema.module if ema is not None else model
            if hasattr(eval_model, "set_flow_strength"):
                eval_model.set_flow_strength(flow_strength)
            records = evaluate_model(
                eval_model, eval_suite, num_samples=eval_num_samples, sample_chunk_size=eval_sample_chunk_size
            )
            summary = summarize_records(records)
            metrics = dict(summary["overall"])
            metrics["step"] = step
            selection_score = metrics.get("cross_entropy_p_to_q_per_dim", metrics.get("m1", float("inf")) + metrics.get("m2", 0.0))
            print(
                f"[eval] step={step} ce/d={metrics.get('cross_entropy_p_to_q_per_dim', float('nan')):.4f} "
                f"m1={metrics['m1']:.4f} m2={metrics['m2']:.4f} "
                f"sw2={metrics.get('sliced_w2', float('nan')):.4f} gap={metrics['energy_gap']:.4f}"
            )
            xfam_summary = x_family_eval_summary(summary)
            if xfam_summary:
                print(f"[eval:x_family] step={step} {xfam_summary}")
            if wandb_run is not None:
                eval_log = {f"eval/{k}": v for k, v in filter_metrics(metrics).items()}
                eval_log.update(x_family_eval_log("eval", summary))
                wandb_run.log(eval_log, step=step)
            if run_dir is not None and selection_score < best_score:
                best_score = selection_score
                save_checkpoint(
                    Path(run_dir) / "best.pt", model, ema, optimizer, scheduler, step, exp_type,
                    metrics=filter_metrics(metrics), extra={"best_score": best_score},
                )
                print(f"[ckpt] step={step} saved best (ce/d={best_score:.4f})")
            train_model.train()
        if use_ddp and eval_due:
            distributed_barrier()

        # ---- Save latest
        save_due = step % save_every == 0 or step == num_steps - 1
        if use_ddp and save_due:
            distributed_barrier()
        if is_main and run_dir is not None and save_due:
            save_checkpoint(
                Path(run_dir) / "latest.pt", model, ema, optimizer, scheduler, step, exp_type,
                metrics=None if metrics is None else filter_metrics(metrics),
                extra={"best_score": best_score},
            )
        if use_ddp and save_due:
            distributed_barrier()

    return {
        "model": model,
        "ema_model": None if ema is None else ema.module,
        "ema": ema,
        "best_score": best_score,
    }
