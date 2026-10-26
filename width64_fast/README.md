# Width-64 Receiver with Frozen DenoiseNN

This is the complete width-64 implementation used by the current production run. It preserves the receiver algorithm, equations, loss, optimizer, precision, and training hyperparameters. DenoiseNN loads the included pretrained asset and remains frozen.

The execution profile uses an effective batch of 112, generation batches of 2, model microbatches of 8, and RE message chunks of 8192. Projection uses eager execution. Signal generation optimizations change indexing and data movement and capture the original ATen geometry operations with CUDA Graphs.

## Files and measurements

- `bayesian_eqdeeprx/`: the complete standalone receiver, physical signal generator, training pipeline, and fixed DenoiseNN asset.
- `RUN_PROFILE.json`: the validated execution profile.
- `CODE_MANIFEST.json`: file hashes and historical checkpoint preparation records.
- `ACCEPTANCE.md`: the original numerical equivalence, throughput, and memory validation report.

The validated mixed-scene measurement was **10.67 seconds per step**, or approximately 675 steps per two hours. Reweighting the measured cases by their configured probabilities gave **10.30 seconds per step**, or approximately 699 steps per two hours. These are hardware-specific measurements, not a guaranteed rate for every window or scene. Cold startup also rebuilds and warms CUDA Graphs.

## Runtime and training

The default Python executable is `D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe`. The production run directory is `D:\BayesianEqDeepRxRuns\width64_fast_batch112`; checkpoints, logs, and caches remain there. The production run originally resumed from a complete 282-step checkpoint toward a total of 70,000 steps.

Run from this version's directory. Use `-PythonPath` and `-RunDir` when the local environment differs.

```powershell
Set-Location .\width64_fast
# Resume a stopped run from its latest complete checkpoint:
& .\bayesian_eqdeeprx\train.ps1 -Resume
```

The checkpoint restores model parameters, LAMB, GradScaler, Python/Torch/CUDA random states, learning-rate progress, and training history. Confirm that the existing worker has stopped before launching a resume command.

For a new run, use a separate directory and complete the matching CUDA preflight first:

```powershell
& .\bayesian_eqdeeprx\train.ps1 -Preflight -Repeats 3 -RunDir 'D:\BayesianEqDeepRxRuns\my_width64_run'
& .\bayesian_eqdeeprx\train.ps1 -RunDir 'D:\BayesianEqDeepRxRuns\my_width64_run'
```

The fixed `pretrained_denoiser.pt` is included because it is required to instantiate the model. Production checkpoints and live logs are not included. On the original training machine, `training.jsonl`, `launch.json`, and `supervision_latest.json` in the run directory record live state; the two-hour monitor is configured separately from this repository.

## Historical records

`ACCEPTANCE.md`, `CODE_MANIFEST.json`, and `RUN_PROFILE.json` describe the original delivery and preflight. Their references to a prepared 282-step checkpoint, training not yet started, or paused monitors are historical. Consult the external run directory for current status.

Source fingerprint: `6e540e10b99014ad718ba57a488e684ee122320a72d55d09641803cbc0310b89`.
