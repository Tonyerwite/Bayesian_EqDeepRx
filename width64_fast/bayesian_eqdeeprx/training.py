"""Reference-scene sampling and joint training with a frozen DenoiseNN."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import BayesianEqDeepRxConfig

from .loss import receiver_loss
from .model import BayesianEqDeepRx
from .provenance import require_source_unchanged, source_fingerprint


class Lamb(torch.optim.Optimizer):
    """LAMB update copied from the completed EqDeepRx training implementation."""

    def __init__(self, params, *, lr: float = 4.4e-3, betas=(0.9, 0.999), eps: float = 1e-6, weight_decay: float = 0.0, bias_correction: bool = True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bool(bias_correction)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                grad = parameter.grad
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                if group["bias_correction"]:
                    bias1 = 1.0 - beta1 ** state["step"]
                    bias2 = 1.0 - beta2 ** state["step"]
                    numerator = state["exp_avg"] / bias1
                    denominator = (state["exp_avg_sq"] / bias2).sqrt()
                else:
                    numerator = state["exp_avg"]
                    denominator = state["exp_avg_sq"].sqrt()
                update = numerator / denominator.add(group["eps"])
                if group["weight_decay"]:
                    update = update + group["weight_decay"] * parameter
                weight_norm = torch.linalg.vector_norm(parameter)
                update_norm = torch.linalg.vector_norm(update)
                if weight_norm == 0 or update_norm == 0 or not torch.isfinite(weight_norm) or not torch.isfinite(update_norm):
                    trust_ratio = 1.0
                else:
                    trust_ratio = float(weight_norm / update_norm)
                parameter.add_(update, alpha=-group["lr"] * trust_ratio)
        return loss


def paper_learning_rate(step: int, *, total_steps: int, base_lr: float, warmup_steps: int = 0) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if step >= total_steps - 1:
        return 0.0
    decay_start = max(0, warmup_steps)
    progress = float(step - decay_start) / float(max(1, total_steps - decay_start - 1))
    return max(0.0, base_lr * (1.0 - progress))


def sample_reference_batch(
    system,
    config: BayesianEqDeepRxConfig,
    *,
    seed: int,
    batch_size: int = 1,
):
    """Call the EqDeepRx Sionna generator with its paper-aligned choices."""

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    random_state = random.Random(seed)
    probabilities = config.training.layer_sampling_probabilities
    if all(p == 1.0 / len(config.layer_counts) for p in probabilities):
        layers = random_state.choice(config.layer_counts)
    else:
        layers = random_state.choices(config.layer_counts, weights=probabilities, k=1)[0]
    if config.training.pilot_sampling_probability_two == 0.5:
        pilot_count = random_state.choice((1, 2))
    else:
        pilot_count = 2 if random_state.random() < config.training.pilot_sampling_probability_two else 1
    interference = random_state.random() < config.interference_probability
    snr_db = torch.tensor(
        [random_state.uniform(*config.snr_db_range) for _ in range(batch_size)],
        dtype=torch.float32,
    )
    return system.generate_batch(
        batch_size=batch_size,
        n_layers=layers,
        pilot_count=pilot_count,
        snr_db=snr_db,
        seed=seed,
        add_interference=interference,
        channel_model="UMa",
        return_true_channel=False,
        coded=False,
    )


def make_optimizer(model: BayesianEqDeepRx, *, learning_rate: float | None = None) -> Lamb:
    """Use reference LAMB settings on the jointly trainable receiver modules."""

    settings = model.config.training
    return Lamb(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings.learning_rate if learning_rate is None else learning_rate,
        betas=(settings.lamb_beta1, settings.lamb_beta2),
        eps=settings.lamb_eps,
        weight_decay=settings.weight_decay,
        bias_correction=settings.lamb_bias_correction,
    )


def uncoded_ber(llrs: torch.Tensor, bits: torch.Tensor, data_mask: torch.Tensor) -> float:
    """Count hard bit errors on data REs using bit-zero-positive LLRs."""

    n, nk, _, f, s = llrs.shape
    if data_mask.shape == (n, 1, f, s):
        data_mask = data_mask.expand(n, nk, f, s)
    selected = data_mask[:, :, None].expand_as(llrs) > 0
    if not bool(selected.any()):
        raise ValueError("BER requires data REs")
    wrong = ((llrs < 0) != (bits > 0.5)) & selected
    return float(wrong.sum().item() / selected.sum().item())


def train_step(model: BayesianEqDeepRx, batch, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Perform exactly one optimizer step on one already generated batch."""

    device = next(model.parameters()).device
    received = batch.received.to(device)
    pilot_symbols = batch.pilot_symbols.to(device)
    pilot_mask = batch.pilot_mask.to(device)
    data_mask = batch.data_mask.to(device)
    bits = batch.target_bits.to(device)
    symbols = batch.transmitted.to(device)
    snr_db = torch.as_tensor(batch.snr_db, device=device, dtype=torch.float32)
    snr_linear = torch.pow(10.0, snr_db / 10.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    final, auxiliary = model(
        received, pilot_symbols, pilot_mask, data_mask, return_aux=True
    )
    objective = receiver_loss(
        auxiliary["readouts"], auxiliary["soft_symbols"], bits,
        symbols, data_mask, snr_linear,
    )
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("receiver loss is non-finite")
    objective.backward()
    optimizer.step()
    return {
        "loss": float(objective.detach().item()),
        "ber": uncoded_ber(final.detach(), bits, data_mask),
    }


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def migrate_training_checkpoint(
    source_path: Path, destination_path: Path, *,
    previous_source_fingerprint: str, config: BayesianEqDeepRxConfig,
    model: BayesianEqDeepRx, new_batch_size: int,
    new_generation_batch_size: int, new_seed: int,
    new_model_microbatch_size: int | None = None,
    new_factor_gather: bool = False,
    new_geometry_graphs: bool = False,
) -> None:
    """Preserve every trained tensor and optimizer moment when resizing a run.

    The source checkpoint remains untouched. The new run records the change
    in its checkpoint and uses a distinct channel seed stream.
    """

    if new_batch_size < 1 or new_generation_batch_size < 1 or new_batch_size % new_generation_batch_size:
        raise ValueError("new generation batch must divide the effective batch")
    new_model_microbatch_size = (new_generation_batch_size if new_model_microbatch_size is None
                                else int(new_model_microbatch_size))
    if (new_model_microbatch_size < new_generation_batch_size
            or new_model_microbatch_size % new_generation_batch_size
            or new_batch_size % new_model_microbatch_size):
        raise ValueError("new_model_microbatch_size must group whole generation batches and divide the effective batch")
    source_path, destination_path = Path(source_path), Path(destination_path)
    if destination_path.exists():
        raise FileExistsError("migration destination already exists")
    saved = torch.load(source_path, map_location="cpu", weights_only=True)
    old_signature = saved.get("run_signature", {})
    step = saved.get("next_step")
    if (saved.get("checkpoint_status") != "complete"
            or saved.get("checkpoint_schema_version") != 1
            or saved.get("source_fingerprint") != previous_source_fingerprint
            or saved.get("config_fingerprint") != _config_fingerprint(config)
            or not isinstance(step, int) or step < 1
            or saved.get("history", {}).get("steps") != step
            or not isinstance(old_signature, dict)
            or old_signature.get("batch_size", 0) < 1
            or old_signature.get("device_type") not in ("cpu", "cuda")):
        raise ValueError("source checkpoint provenance, configuration or step is invalid")
    if any(len(saved["history"].get(key, [])) != step
           for key in ("losses", "bers", "learning_rates")):
        raise ValueError("source checkpoint history is incomplete")
    current = model.state_dict()
    checkpoint_state = saved["model_state_dict"]
    if any(name not in checkpoint_state or not torch.equal(tensor, checkpoint_state[name])
           for name, tensor in current.items() if name.startswith("front_end.denoise.")):
        raise ValueError("source checkpoint frozen DenoiseNN does not match the asset")
    model.load_state_dict(checkpoint_state, strict=True)
    optimizer = make_optimizer(model)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    errors = _state_errors(model, optimizer, step)
    expected_lr = paper_learning_rate(step - 1, total_steps=config.training.total_steps,
                                      base_lr=config.training.learning_rate,
                                      warmup_steps=config.training.warmup_steps)
    if errors or any(not math.isclose(float(group["lr"]), expected_lr, rel_tol=1e-12, abs_tol=1e-15)
                     for group in optimizer.param_groups):
        raise ValueError("source checkpoint optimizer is inconsistent: " + ", ".join(errors))
    migrated = copy.deepcopy(saved)
    migrated["source_fingerprint"] = require_source_unchanged()
    migrated["run_signature"] = {
        **old_signature, "batch_size": new_batch_size,
        "generation_batch_size": new_generation_batch_size, "seed": int(new_seed),
        **_execution_signature(model, new_model_microbatch_size, new_factor_gather, new_geometry_graphs),
    }
    migrated["batch_migration"] = {
        "previous_batch_size": old_signature["batch_size"],
        "previous_generation_batch_size": old_signature["generation_batch_size"],
        "previous_source_fingerprint": previous_source_fingerprint,
        "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "migration_step": step,
        "previous_execution_settings": {name: old_signature.get(name) for name in (
            "model_microbatch_size", "message_chunk_size", "factor_gather", "geometry_graphs")},
    }
    atomic_torch_save(migrated, destination_path)


def _config_fingerprint(config: BayesianEqDeepRxConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _execution_signature(model, model_microbatch_size, factor_gather=False, geometry_graphs=False):
    return {
        "model_microbatch_size": model_microbatch_size,
        "message_chunk_size": getattr(model, 'message_chunk_size', None),
        "factor_gather": bool(factor_gather),
        "geometry_graphs": bool(geometry_graphs),
    }


def _state_errors(model, optimizer, expected_step: int) -> list[str]:
    errors = []
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all().item():
            errors.append("model." + name)
    trainable = {parameter for parameter in model.parameters() if parameter.requires_grad}
    if set(optimizer.state) != trainable:
        errors.append("optimizer parameter coverage")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter in optimizer.state:
                errors.append("frozen parameter in optimizer: " + name)
            continue
        state = optimizer.state.get(parameter)
        if state is None or state.get("step") != expected_step:
            errors.append("optimizer step: " + name)
        elif any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item() for value in state.values()):
            errors.append("optimizer non-finite: " + name)
    return errors


def _sample_case(config, random_state, *, n_layers, pilot_count, interference_present):
    if n_layers is None:
        probabilities = config.training.layer_sampling_probabilities
        layers = (random_state.choice(config.layer_counts)
                  if all(p == 1.0 / len(config.layer_counts) for p in probabilities)
                  else random_state.choices(config.layer_counts, weights=probabilities, k=1)[0])
    else:
        layers = n_layers
    config.validate_layer_count(layers)
    if pilot_count is None:
        pilots = (random_state.choice((1, 2))
                  if config.training.pilot_sampling_probability_two == 0.5
                  else 2 if random_state.random() < config.training.pilot_sampling_probability_two else 1)
    else:
        pilots = pilot_count
    if pilots not in (1, 2):
        raise ValueError("pilot_count must be 1 or 2")
    sampled = random_state.random() < config.interference_probability
    interference = sampled if interference_present is None else bool(interference_present)
    return layers, pilots, interference


def train_steps(
    model, system, config: BayesianEqDeepRxConfig, *, steps: int,
    batch_size: int | None = None, generation_batch_size: int = 2,
    model_microbatch_size: int | None = None,
    n_layers: int | None = None, pilot_count: int | None = None,
    interference_present: bool | None = None, seed: int | None = None,
    output_path: Path | None = None, resume_path: Path | None = None,
    save_every: int = 1, device: torch.device | str = "cpu",
    progress_callback=None,
) -> dict:
    """Train all unfrozen modules using the reference case and SNR sampling.

    Every generation batch contributes its sample fraction to one effective
    batch update. There is no cross-sample normalization in this receiver.
    """

    if steps < 1 or save_every < 1:
        raise ValueError("steps and save_every must be positive")
    batch_size = config.training.batch_size if batch_size is None else int(batch_size)
    if batch_size < 1 or generation_batch_size < 1 or batch_size % generation_batch_size:
        raise ValueError("generation_batch_size must divide the effective batch")
    model_microbatch_size = (generation_batch_size if model_microbatch_size is None
                            else int(model_microbatch_size))
    if (model_microbatch_size < generation_batch_size
            or model_microbatch_size % generation_batch_size
            or batch_size % model_microbatch_size):
        raise ValueError("model_microbatch_size must group whole generation batches and divide the effective batch")
    source_hash = require_source_unchanged()
    device = torch.device(device)
    model.to(device)
    seed = config.training.seed if seed is None else int(seed)
    random_state = random.Random(seed)
    amp_enabled = device.type == "cuda" and config.training.amp_dtype == "bfloat16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=1.0)
    optimizer = make_optimizer(model)
    signature = {
        "batch_size": batch_size, "generation_batch_size": generation_batch_size,
        "seed": seed, "n_layers": n_layers, "pilot_count": pilot_count,
        "interference_present": interference_present,
        "device_type": device.type,
        **_execution_signature(model, model_microbatch_size, getattr(system, 'factor_gather', False),
                               getattr(system, 'geometry_graphs', False)),
    }
    case_counts = {f"{layers}:{pilots}:{int(interference)}": 0
                   for layers in config.layer_counts for pilots in (1, 2)
                   for interference in (False, True)}
    history = {"steps": 0, "losses": [], "bers": [], "learning_rates": [],
               "training_case_step_counts": case_counts}
    start_step = 0
    batch_migration = None
    if resume_path is not None:
        saved = torch.load(Path(resume_path), map_location=device, weights_only=True)
        if (saved.get("checkpoint_status") != "complete"
                or saved.get("checkpoint_schema_version") != 1
                or saved.get("source_fingerprint") != source_hash
                or saved.get("config_fingerprint") != _config_fingerprint(config)
                or saved.get("run_signature") != signature):
            raise ValueError("resume checkpoint provenance, configuration or run signature does not match")
        start_step = saved.get("next_step")
        restored_history = saved.get("history")
        if (not isinstance(start_step, int) or start_step < 0 or start_step > steps
                or not isinstance(restored_history, dict)
                or restored_history.get("steps") != start_step
                or any(len(restored_history.get(key, [])) != start_step
                       for key in ("losses", "bers", "learning_rates"))
                or sum(restored_history.get("training_case_step_counts", {}).values()) != start_step):
            raise ValueError("resume checkpoint step and history are inconsistent")
        if any(not math.isfinite(float(value)) for key in ("losses", "bers", "learning_rates")
               for value in restored_history[key]):
            raise ValueError("resume checkpoint history contains non-finite values")
        saved_groups = saved["optimizer_state_dict"].get("param_groups", [])
        expected_groups = optimizer.param_groups
        prior_lr = (paper_learning_rate(start_step - 1,
                    total_steps=config.training.total_steps,
                    base_lr=config.training.learning_rate,
                    warmup_steps=config.training.warmup_steps)
                    if start_step else config.training.learning_rate)
        if len(saved_groups) != len(expected_groups) or any(
            len(actual.get("params", [])) != len(expected["params"])
            or tuple(actual.get("betas", ())) != tuple(expected["betas"])
            or actual.get("eps") != expected["eps"]
            or actual.get("weight_decay") != expected["weight_decay"]
            or actual.get("bias_correction") != expected["bias_correction"]
            or not math.isclose(float(actual.get("lr", float("nan"))), prior_lr,
                                rel_tol=1e-12, abs_tol=1e-15)
            for actual, expected in zip(saved_groups, expected_groups)
        ):
            raise ValueError("resume checkpoint optimizer hyperparameters or learning rate are inconsistent")
        current_state = model.state_dict()
        saved_state = saved["model_state_dict"]
        if any(name not in saved_state or not torch.equal(value, saved_state[name])
               for name, value in current_state.items()
               if name.startswith("front_end.denoise.")):
            raise ValueError("resume checkpoint frozen DenoiseNN differs from the bundled reference weights")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        errors = _state_errors(model, optimizer, start_step) if start_step else []
        if errors:
            raise ValueError("resume checkpoint optimizer is inconsistent: " + ", ".join(errors))
        scaler.load_state_dict(saved["grad_scaler_state_dict"])
        random_state.setstate(saved["python_random_state"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state_all"]])
        history = copy.deepcopy(restored_history)
        batch_migration = copy.deepcopy(saved.get("batch_migration"))

    for step in range(start_step, steps):
        if source_fingerprint() != source_hash:
            raise RuntimeError("training source changed; restart from the last complete checkpoint")
        started = time.perf_counter()
        layers, pilots, interference = _sample_case(
            config, random_state, n_layers=n_layers, pilot_count=pilot_count,
            interference_present=interference_present,
        )
        optimizer.zero_grad(set_to_none=True)
        model.train()
        step_loss = 0.0
        step_ber = 0.0
        pending = []
        for generated in range(0, batch_size, generation_batch_size):
            snr_db = torch.tensor(
                [random_state.uniform(*config.snr_db_range)
                 for _ in range(generation_batch_size)], dtype=torch.float32,
            )
            batch = system.generate_batch(
                batch_size=generation_batch_size, n_layers=layers,
                pilot_count=pilots, snr_db=snr_db,
                seed=seed + step * batch_size + generated,
                add_interference=interference, channel_model="UMa",
                return_true_channel=False, coded=False,
            )
            if steps > 100 and batch.backend != "sionna_tr38901_time_domain":
                raise RuntimeError("long training requires the reference time-domain Sionna backend")
            snr_values = torch.as_tensor(batch.snr_db, device=device, dtype=torch.float32).flatten()
            if snr_values.numel() == 1:
                snr_values = snr_values.expand(generation_batch_size)
            pending.append((batch.received.to(device), batch.pilot_symbols.to(device),
                batch.pilot_mask.to(device), batch.data_mask.to(device),
                batch.target_bits.to(device), batch.transmitted.to(device), snr_values))
            del batch
            if len(pending) * generation_batch_size < model_microbatch_size:
                continue
            received, pilot_symbols, pilot_mask, data_mask, bits, symbols, snr_values = (
                pending[0][index] if len(pending) == 1 else
                torch.cat([part[index] for part in pending], dim=0)
                for index in range(7))
            pending.clear()
            snr_linear = torch.pow(10.0, snr_values / 10.0)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                final, auxiliary = model(received, pilot_symbols, pilot_mask, data_mask, return_aux=True)
                objective = receiver_loss(auxiliary["readouts"], auxiliary["soft_symbols"],
                                          bits, symbols, data_mask, snr_linear)
                weighted = objective * (model_microbatch_size / batch_size)
            if not torch.isfinite(weighted.detach()).all().item():
                raise FloatingPointError(f"non-finite Bayesian loss in optimizer step {step}")
            scaler.scale(weighted).backward()
            step_loss += float(weighted.detach())
            step_ber += (model_microbatch_size / batch_size) * uncoded_ber(final.detach(), bits, data_mask)
            del received, pilot_symbols, pilot_mask, data_mask, bits, symbols, final, auxiliary, objective, weighted
        if amp_enabled:
            scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(f"missing or non-finite gradient at step {step}: {name}")
        lr = paper_learning_rate(step, total_steps=config.training.total_steps,
                                 base_lr=config.training.learning_rate,
                                 warmup_steps=config.training.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        completed = step + 1
        errors = _state_errors(model, optimizer, completed)
        if errors or not math.isfinite(step_loss) or not math.isfinite(step_ber):
            raise FloatingPointError("inconsistent joint update: " + ", ".join(errors))
        history["steps"] = completed
        history["losses"].append(step_loss)
        history["bers"].append(step_ber)
        history["learning_rates"].append(lr)
        history["training_case_step_counts"][f"{layers}:{pilots}:{int(interference)}"] += 1
        optimizer.zero_grad(set_to_none=True)
        if output_path is not None and (completed % save_every == 0 or completed == steps):
            require_source_unchanged()
            checkpoint = {
                "checkpoint_status": "complete", "checkpoint_schema_version": 1,
                "source_fingerprint": source_hash,
                "config_fingerprint": _config_fingerprint(config),
                "run_signature": signature, "next_step": completed,
                "model_state_dict": _cpu_copy(model.state_dict()),
                "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
                "grad_scaler_state_dict": _cpu_copy(scaler.state_dict()),
                "history": _cpu_copy(history),
                "python_random_state": random_state.getstate(),
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_state_all": [state.clone() for state in torch.cuda.get_rng_state_all()] if device.type == "cuda" else [],
            }
            if batch_migration is not None:
                checkpoint["batch_migration"] = copy.deepcopy(batch_migration)
            atomic_torch_save(checkpoint, Path(output_path))
        if progress_callback is not None:
            progress_callback({
                "step": completed, "loss": step_loss, "ber": step_ber,
                "learning_rate": lr, "layers": layers, "pilot_count": pilots,
                "interference_present": interference,
                "step_seconds": time.perf_counter() - started,
                "module_optimizer_steps": {
                    prefix: sorted({optimizer.state[p]["step"] for name, p in model.named_parameters()
                                    if name.startswith(prefix) and p.requires_grad})
                    for prefix in ("state_init.", "updates.", "demapper.", "log_sigma")
                },
            })
    return history
