"""CUDA preflight and resumable long training for bayesian_eqdeeprx."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="bayesian_eqdeeprx reference-scene training")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "migrate", "train"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=112)
        command.add_argument("--cpu-threads", type=int, default=4)
        if name == "preflight":
            command.add_argument("--generation-batch-size", type=int, default=2)
            command.add_argument("--message-chunk-size", type=int, default=64)
            command.add_argument("--model-microbatch-size", type=int)
            command.add_argument("--factor-gather", action="store_true")
            command.add_argument("--geometry-graphs", action="store_true")
            command.add_argument("--repeats", type=int, default=2)
        elif name == "migrate":
            command.add_argument("--source-run-dir", type=Path, required=True)
            command.add_argument("--generation-batch-size", type=int, default=2)
            command.add_argument("--seed", type=int, default=1_002_026)
        else:
            command.add_argument("--resume", action="store_true")
            command.add_argument("--seed", type=int, default=2026)
            command.add_argument("--steps", type=int, default=70_000)
            command.add_argument("--save-every", type=int, default=1)
            command.add_argument("--confirm", action="store_true")
    return parser


def _prepare_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("long training requires the CUDA environment used by EqDeepRx")
    torch.cuda.set_per_process_memory_fraction(0.85, torch.cuda.current_device())
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda")


def _log(event: dict, path: Path | None = None) -> None:
    line = json.dumps(event, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def preflight(args, device):
    import torch
    from .config import BayesianEqDeepRxConfig
    from .model import BayesianEqDeepRx
    from .provenance import require_source_unchanged
    from .sionna_system import SionnaTR38901System
    from .training import _config_fingerprint, train_steps

    config = BayesianEqDeepRxConfig()
    if args.batch_size < 1 or args.generation_batch_size < 1 or args.batch_size % args.generation_batch_size:
        raise ValueError("generation batch must divide the effective batch")
    if args.repeats < 2:
        raise ValueError("preflight requires a warmup and at least one measured step")
    microbatch = (args.generation_batch_size if args.model_microbatch_size is None
                  else args.model_microbatch_size)
    if (microbatch < args.generation_batch_size or microbatch % args.generation_batch_size
            or args.batch_size % microbatch):
        raise ValueError('model_microbatch_size must group whole generation batches and divide the effective batch')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    measured_batch = args.batch_size if microbatch > args.generation_batch_size else args.generation_batch_size
    system = SionnaTR38901System(config, device, factor_gather=args.factor_gather,
                               geometry_graphs=args.geometry_graphs)
    cases = []
    total_memory = torch.cuda.get_device_properties(device).total_memory
    for layers in config.layer_counts:
        for pilots in (1, 2):
            for interference in (False, True):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(config.training.seed)
                receiver = BayesianEqDeepRx(config, message_chunk_size=args.message_chunk_size).to(
                    device, memory_format=torch.channels_last
                )
                events = []
                train_steps(
                    receiver, system, config, steps=args.repeats, batch_size=measured_batch,
                    generation_batch_size=args.generation_batch_size,
                    model_microbatch_size=microbatch,
                    n_layers=layers, pilot_count=pilots, interference_present=interference,
                    device=device, progress_callback=events.append,
                )
                record = {
                    "layers": layers, "pilot_count": pilots,
                    "interference_present": interference,
                    "seconds_per_measured_update": statistics.median(
                        event["step_seconds"] for event in events[1:]
                    ),
                    "loss": events[-1]["loss"], "ber": events[-1]["ber"],
                    "module_optimizer_steps": events[-1]["module_optimizer_steps"],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                cases.append(record)
                _log(record)
                del receiver
    mean_seconds = 0.0
    for case in cases:
        i = config.layer_counts.index(case["layers"])
        probability = config.training.layer_sampling_probabilities[i]
        probability *= (config.training.pilot_sampling_probability_two if case["pilot_count"] == 2
                        else 1 - config.training.pilot_sampling_probability_two)
        probability *= (config.interference_probability if case["interference_present"]
                        else 1 - config.interference_probability)
        mean_seconds += probability * case["seconds_per_measured_update"]
    estimated_step = mean_seconds * args.batch_size / measured_batch
    memory_safe = max(case["peak_reserved_bytes"] for case in cases) <= 0.85 * total_memory
    report = {
        "long_training_ready": memory_safe,
        "source_fingerprint": require_source_unchanged(),
        "config_fingerprint": _config_fingerprint(config),
        "torch_version": torch.__version__,
        "sionna_version": importlib.metadata.version("sionna"),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_bytes": total_memory,
        "cuda_memory_fraction": torch.cuda.get_per_process_memory_fraction(),
        "cuda_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "effective_batch_size": args.batch_size,
        "generation_batch_size": args.generation_batch_size,
        "message_chunk_size": args.message_chunk_size,
        "model_microbatch_size": microbatch, "factor_gather": args.factor_gather,
        "geometry_graphs": args.geometry_graphs,
        "measurement_batch_size": measured_batch, "cpu_threads": torch.get_num_threads(),
        "estimated_optimizer_step_seconds": estimated_step,
        "estimated_total_training_days": estimated_step * config.training.total_steps / 86_400,
        "cases": cases,
    }
    _atomic_json(report, args.run_dir / "preflight.json")
    _log({key: value for key, value in report.items() if key != "cases"})
    if not memory_safe:
        raise RuntimeError("preflight exceeds the reserved 15% GPU memory margin")
    return report


def train(args, device):
    import torch
    from .config import BayesianEqDeepRxConfig
    from .model import BayesianEqDeepRx
    from .provenance import require_source_unchanged
    from .sionna_system import SionnaTR38901System
    from .training import _config_fingerprint, train_steps

    config = BayesianEqDeepRxConfig()
    if args.steps not in (417, 418, config.training.total_steps) or not args.confirm:
        raise ValueError("width-64 trial requires --steps 417, 418 or 70000 and --confirm")
    report = json.loads((args.run_dir / "preflight.json").read_text(encoding="utf-8"))
    expected = {
        "long_training_ready": True,
        "source_fingerprint": require_source_unchanged(),
        "config_fingerprint": _config_fingerprint(config),
        "torch_version": torch.__version__,
        "sionna_version": importlib.metadata.version("sionna"),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cuda_memory_fraction": torch.cuda.get_per_process_memory_fraction(),
        "cuda_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "effective_batch_size": args.batch_size,
        "cpu_threads": torch.get_num_threads(),
    }
    wrong = [name for name, value in expected.items() if report.get(name) != value]
    if wrong:
        raise ValueError("preflight no longer matches training runtime: " + ", ".join(wrong))
    checkpoint = args.run_dir / "bayesian_eqdeeprx.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError("checkpoint already exists; use --resume")
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError("no complete checkpoint to resume")
    torch.manual_seed(config.training.seed)
    receiver = BayesianEqDeepRx(config, message_chunk_size=report["message_chunk_size"]).to(
        device, memory_format=torch.channels_last
    )
    system = SionnaTR38901System(config, device, factor_gather=report.get('factor_gather', False),
                               geometry_graphs=report.get('geometry_graphs', False))
    log = args.run_dir / "training.jsonl"
    _log({"event": "training_start", "resume": args.resume,
          "checkpoint": str(checkpoint), "target_steps": args.steps,
          "effective_batch_size": args.batch_size,
          "generation_batch_size": report["generation_batch_size"]}, log)
    try:
        history = train_steps(
            receiver, system, config, steps=args.steps,
            batch_size=args.batch_size, seed=args.seed,
            generation_batch_size=report["generation_batch_size"],
            model_microbatch_size=report.get('model_microbatch_size'),
            output_path=checkpoint, resume_path=checkpoint if args.resume else None,
            save_every=args.save_every, device=device,
            progress_callback=lambda event: _log(event, log),
        )
    except BaseException as error:
        _log({"event": "training_stopped", "error": repr(error),
              "resume_from": str(checkpoint)}, log)
        raise
    _log({"event": "training_complete", "steps": history["steps"]}, log)
    return history


def migrate(args, device):
    from .config import BayesianEqDeepRxConfig
    from .model import BayesianEqDeepRx
    from .provenance import require_source_unchanged
    from .training import _config_fingerprint, migrate_training_checkpoint

    config = BayesianEqDeepRxConfig()
    old_report = json.loads((args.source_run_dir / "preflight.json").read_text(encoding="utf-8"))
    new_report = json.loads((args.run_dir / "preflight.json").read_text(encoding="utf-8"))
    if (old_report.get("config_fingerprint") != _config_fingerprint(config)
            or new_report.get("config_fingerprint") != _config_fingerprint(config)
            or new_report.get("source_fingerprint") != require_source_unchanged()
            or new_report.get("effective_batch_size") != args.batch_size
            or new_report.get("generation_batch_size") != args.generation_batch_size
            or new_report.get("long_training_ready") is not True):
        raise ValueError("source or destination preflight does not match this migration")
    model = BayesianEqDeepRx(config, message_chunk_size=new_report["message_chunk_size"])
    source = args.source_run_dir / "bayesian_eqdeeprx.pt"
    destination = args.run_dir / "bayesian_eqdeeprx.pt"
    migrate_training_checkpoint(
        source, destination,
        previous_source_fingerprint=old_report["source_fingerprint"],
        config=config, model=model,
        new_batch_size=args.batch_size,
        new_generation_batch_size=args.generation_batch_size,
        new_model_microbatch_size=new_report.get('model_microbatch_size'),
        new_factor_gather=new_report.get('factor_gather', False),
        new_geometry_graphs=new_report.get('geometry_graphs', False),
        new_seed=args.seed,
    )
    _log({"event": "batch_migration", "source": str(source),
          "destination": str(destination), "batch_size": args.batch_size,
          "seed": args.seed})


def main() -> None:
    args = build_parser().parse_args()
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch._dynamo.config.recompile_limit = 32
    device = _prepare_cuda()
    {"preflight": preflight, "migrate": migrate, "train": train}[args.command](args, device)


if __name__ == "__main__":
    main()
