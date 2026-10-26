"""Independent full-joint training and CPU validation; no automatic long run."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "smoke"):
        p = commands.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--cpu-threads", type=int, default=1 if name == "smoke" else 4)
        p.add_argument("--batch-size", type=int, default=1 if name == "smoke" else 112)
        p.add_argument("--generation-batch-size", type=int, default=1 if name == "smoke" else 2)
        p.add_argument("--model-microbatch-size", type=int, default=1 if name == "smoke" else 2)
        p.add_argument("--message-chunk-size", type=int, default=128 if name == "smoke" else 8192)
        p.add_argument("--factor-gather", action="store_true")
        p.add_argument("--geometry-graphs", action="store_true")
        if name == "train":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--seed", type=int, default=2026)
            p.add_argument("--steps", type=int, choices=(70000,), default=70000)
            p.add_argument("--save-every", type=int, default=1)
            p.add_argument("--confirm", action="store_true")
        else:
            p.add_argument("--repeats", type=int, default=1 if name == "smoke" else 2)
    return parser


def require_exclusive_cuda():
    """Query before allocating any CUDA memory; never compete with a running job."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot verify exclusive GPU access") from error
    others = {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
    others.discard(os.getpid())
    if others:
        raise RuntimeError("GPU already in use by process(es): " + str(sorted(others)))


def _prepare_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Formal preflight/training requires CUDA; use smoke for CPU validation")
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda")


def _log(event, path=None):
    line = json.dumps(event, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _atomic_json(payload, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _runtime_contract(args, config, device):
    import torch
    from .provenance import require_source_unchanged
    from .training import _config_fingerprint
    return {
        "long_training_ready": True, "training_mode": "full_joint", "initialization": "random",
        "source_fingerprint": require_source_unchanged(), "config_fingerprint": _config_fingerprint(config),
        "torch_version": torch.__version__, "sionna_version": importlib.metadata.version("sionna"),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cuda_memory_fraction": torch.cuda.get_per_process_memory_fraction(),
        "cuda_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads(), "effective_batch_size": args.batch_size,
        "generation_batch_size": args.generation_batch_size,
        "model_microbatch_size": args.model_microbatch_size, "message_chunk_size": args.message_chunk_size,
        "factor_gather": args.factor_gather, "geometry_graphs": args.geometry_graphs,
    }


def validate_preflight(report, expected):
    wrong = [key for key, value in expected.items() if report.get(key) != value]
    if wrong:
        raise ValueError("preflight does not match full-joint CUDA training: " + ", ".join(wrong))


def _run_cases(args, device, *, formal):
    import torch
    from .config import BayesianEqDeepRxConfig
    from .model import BayesianEqDeepRx
    from .sionna_system import SionnaTR38901System
    from .training import MODULE_PREFIXES, train_steps
    from .provenance import require_source_unchanged
    if args.repeats < (2 if formal else 1):
        raise ValueError("preflight needs warmup and measured updates; smoke needs at least one update")
    config = BayesianEqDeepRxConfig()
    system = SionnaTR38901System(config, device, factor_gather=args.factor_gather,
                                geometry_graphs=args.geometry_graphs)
    cases = []
    for layers in config.layer_counts:
        for pilots in (1, 2):
            for interference in (False, True):
                if formal:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(config.training.seed)
                receiver = BayesianEqDeepRx(config, message_chunk_size=args.message_chunk_size)
                if formal:
                    receiver = receiver.to(device, memory_format=torch.channels_last)
                initial = {k: v.detach().cpu().clone() for k, v in receiver.state_dict().items()}
                case_dir = args.run_dir / f"K{layers}_DMRS{pilots}_I{int(interference)}"
                events = []
                history = train_steps(receiver, system, config, steps=args.repeats,
                    batch_size=args.batch_size, generation_batch_size=args.generation_batch_size,
                    model_microbatch_size=args.model_microbatch_size, n_layers=layers,
                    pilot_count=pilots, interference_present=interference,
                    device=device, output_path=None if formal else case_dir / "smoke.pt",
                    progress_callback=events.append)
                changed = {prefix: any(not torch.equal(initial[k], v.detach().cpu())
                    for k, v in receiver.state_dict().items() if k.startswith(prefix))
                    for prefix in MODULE_PREFIXES}
                if not all(changed.values()):
                    raise RuntimeError("A trainable module did not update: " + str(changed))
                measured = events[1:] if formal else events
                record = {"layers": layers, "pilot_count": pilots, "interference_present": interference,
                    "updates": history["steps"], "loss": events[-1]["loss"], "ber": events[-1]["ber"],
                    "bit_loss": events[-1]["bit_loss"], "symbol_loss": events[-1]["symbol_loss"],
                    "round_bit_losses": events[-1]["round_bit_losses"], "modules_changed": changed,
                    "module_optimizer_steps": events[-1]["module_optimizer_steps"],
                    "mean_step_seconds": statistics.mean(x["step_seconds"] for x in measured)}
                if formal:
                    record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
                cases.append(record)
                _log(record, args.run_dir / "validation.jsonl")
                del receiver
    if formal:
        report = _runtime_contract(args, config, device)
        report["long_training_ready"] = max(x["peak_reserved_bytes"] for x in cases) <= 0.85 * report["gpu_total_bytes"]
        report["estimated_optimizer_step_seconds"] = statistics.mean(x["mean_step_seconds"] for x in cases)
        report["cases"] = cases
        _atomic_json(report, args.run_dir / "preflight.json")
        if not report["long_training_ready"]:
            raise RuntimeError("Joint preflight exceeds the reserved GPU memory margin")
    else:
        report = {"cpu_smoke_passed": True, "long_training_ready": False,
            "cuda_validated": False, "training_mode": "full_joint", "initialization": "random",
            "source_fingerprint": require_source_unchanged(), "case_count": len(cases),
            "physical_grid": {"subcarriers": 192, "symbols": 14, "rx_antennas": 16},
            "batch_size": args.batch_size, "cases": cases}
        _atomic_json(report, args.run_dir / "cpu_smoke.json")
    _log({k: v for k, v in report.items() if k != "cases"})
    return report


def train(args, device):
    import torch
    from .config import BayesianEqDeepRxConfig
    from .model import BayesianEqDeepRx
    from .sionna_system import SionnaTR38901System
    from .training import train_steps
    if not args.confirm:
        raise ValueError("Formal training requires --confirm")
    config = BayesianEqDeepRxConfig()
    report = json.loads((args.run_dir / "preflight.json").read_text(encoding="utf-8"))
    validate_preflight(report, _runtime_contract(args, config, device))
    checkpoint = args.run_dir / "bayesian_eqdeeprx.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError("checkpoint already exists; use --resume")
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError("no complete joint checkpoint to resume")
    torch.manual_seed(config.training.seed)
    receiver = BayesianEqDeepRx(config, message_chunk_size=args.message_chunk_size).to(
        device, memory_format=torch.channels_last)
    system = SionnaTR38901System(config, device, factor_gather=args.factor_gather,
                               geometry_graphs=args.geometry_graphs)
    log = args.run_dir / "training.jsonl"
    _log({"event": "training_start", "training_mode": "full_joint", "initialization": "random",
          "resume": args.resume, "target_steps": args.steps, "effective_batch_size": args.batch_size}, log)
    try:
        history = train_steps(receiver, system, config, steps=args.steps, batch_size=args.batch_size,
            seed=args.seed, generation_batch_size=args.generation_batch_size,
            model_microbatch_size=args.model_microbatch_size, output_path=checkpoint,
            resume_path=checkpoint if args.resume else None, save_every=args.save_every,
            device=device, progress_callback=lambda event: _log(event, log))
    except BaseException as error:
        _log({"event": "training_stopped", "error": repr(error)}, log)
        raise
    _log({"event": "training_complete", "steps": history["steps"]}, log)
    return history


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "smoke":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if os.name == "nt":
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    elif args.command == "train" and not args.confirm:
        parser.error("Formal training requires --confirm")
    from .training import validate_run_directory
    args.run_dir = validate_run_directory(args.run_dir)
    if args.command != "smoke":
        require_exclusive_cuda()
    import torch
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch._dynamo.config.recompile_limit = 32
    device = torch.device("cpu") if args.command == "smoke" else _prepare_cuda()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        train(args, device)
    else:
        _run_cases(args, device, formal=args.command == "preflight")


if __name__ == "__main__":
    main()
