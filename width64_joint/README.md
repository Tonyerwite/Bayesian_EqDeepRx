# Width-64 Full Joint Training

This independent version randomly initializes and jointly trains DenoiseNN, StateInit, all four StateUpdate networks, DemapperNN, and the four sigma parameters through the end-to-end bit/symbol objective. Formal long training has not been started.

The receiver retains four CNN/belief rounds. Each round performs two directed message updates and one soft projection. The objective averages the four bit cross-entropies and supervises the first two channels before projection with the original data mask, SNR weighting, and `1e-5` symbol weight. It adds neither true-channel supervision nor DetectorNN-specific VCL. Reported `bit_loss` and `symbol_loss` already include their weights; `round_bit_losses` contain the four SNR-weighted cross-entropies.

All trainable parameters use one LAMB optimizer: effective batch 112, generation batch 2, base learning rate 0.0044, and linear decay through 70,000 steps with the reference beta, epsilon, and weight-decay settings. The default model microbatch is **2**. This is a distinct execution profile; the frozen model's microbatch-8 speed and memory results do not establish joint-training performance.

## Validation scope

CPU validation covers gradients and actual updates across the receiver, effective-batch-112 accumulation, strict checkpoint resume, and real time-domain Sionna UMa generation with a 192-by-14 grid, 16 receive antennas, and all 12 combinations of K=2/3/4, one/two DMRS symbols, and external interference absent/present. Test updates use isolated outputs and do not establish convergence or receiver performance.

CUDA BF16 behavior, full-batch GPU memory, throughput, and long-run convergence have not been validated. CPU smoke reports explicitly set `long_training_ready=false` and `cuda_validated=false`. Formal training requires a separate, matching CUDA preflight. A BER improvement over EqDeepRx has not been demonstrated.

## CPU smoke test

The default Python executable is `D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe`. Override `-PythonPath` for another environment.

```powershell
Set-Location .\width64_joint
& .\bayesian_eqdeeprx\train.ps1 -Smoke
```

The smoke command hides CUDA, uses one CPU thread and low process priority, and creates a separate timestamped validation directory on drive D. Each of the 12 scenarios uses a fresh random model, one physical sample, and one test update. It is not a continuous 12-step training run. Results are written to `cpu_smoke.json` and each scenario's `smoke.pt`.

Inspect commands without executing them:

```powershell
& .\bayesian_eqdeeprx\train.ps1 -Smoke -DryRun
& .\bayesian_eqdeeprx\train.ps1 -Preflight -DryRun
& .\bayesian_eqdeeprx\train.ps1 -Train -DryRun
```

## Formal training entry points

The default formal run directory is `D:\BayesianEqDeepRxRuns\width64_joint_batch112`. When GPU resources are available, run preflight before starting training. CUDA entry points reject other GPU compute processes.

```powershell
& .\bayesian_eqdeeprx\train.ps1 -Preflight
& .\bayesian_eqdeeprx\train.ps1 -Train
# Resume only after this version's worker has stopped:
& .\bayesian_eqdeeprx\train.ps1 -Resume
```

Use `-RunDir` and `-PythonPath` to override the local defaults. Omitting the operation is an error and never starts training automatically. Preflight covers all 12 scenarios with effective batch 112 and at least one warmup and one measured update per case.

New training starts from random initialization. Resume accepts only this version's schema-2 complete checkpoints and restores the trained DenoiseNN, full LAMB state, GradScaler, histories, and Python/Torch/CUDA random states. Frozen-version checkpoints cannot be resumed directly. Source-change and output-path guards protect the run and earlier versions.

## Automated tests

Run from this version directory. Tests force one CPU thread and low priority and do not use the GPU. The real Sionna resume test also exercises the physical signal generator.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe' -m pytest tests -q -o cache_dir=D:/BayesianEqDeepRxRuns/width64_joint_validation/pytest-cache --basetemp=D:/BayesianEqDeepRxRuns/width64_joint_validation/pytest_manual
```

The 27 automated tests cover module updates, loss and gradient routing, complete optimizer coverage, effective-batch accumulation, incompatible-checkpoint rejection, entry-point isolation, and exact uninterrupted-versus-resumed CPU state. Validation of the original implementation also covered all 12 physical CPU smoke scenarios.

This package is self-contained. Its physical model, belief equations, CNN topology, and signal generation were copied from the validated width-64 implementation. It does not import or modify the earlier version.

Source fingerprint: `df31da2cb5bfbb7161915c4fc25724b858fb137e5797da089e83a888f4becfe3`.
