# Bayesian EqDeepRx — Width 64

This repository contains two self-contained width-64 receivers and their complete training pipelines.

| Version | Directory | DenoiseNN | Status |
|---|---|---|---|
| Current production version | [width64_fast](width64_fast/README.md) | Loaded from the included pretrained asset and frozen | CUDA profile validated; the 70,000-step run is in progress |
| Full joint training | [width64_joint](width64_joint/README.md) | Randomly initialized and trained end to end | CPU tests, physical Sionna scenarios, and exact resume validated; CUDA preflight and formal long training have not been run |

Both versions include channel estimation, interpolation, LMMSE/RZF, StateInit, four StateUpdate networks, belief messages and soft projection, DemapperNN, losses, Sionna signal generation, LAMB optimization, complete checkpoint resume, and PowerShell entry points.

## Training volume per optimizer step

Both versions use an effective batch of **112 slot samples** and perform one LAMB update per step. Each slot uses a 192-by-14 resource grid and 16 receive antennas. Training samples 2, 3, or 4 users, one or two DMRS symbols, and the presence or absence of an external interferer.

| Setting | Frozen DenoiseNN | Full joint training |
|---|---:|---:|
| Effective batch size | 112 | 112 |
| Signal generation batch size | 2 | 2 |
| Generation calls per optimizer step | 56 | 56 |
| Model microbatch size | 8 | 2 |
| Gradient accumulation passes per optimizer step | 14 | 56 |
| RE message chunk size | 8192 | 8192 |
| Target optimizer steps | 70000 | 70000 |

The four internal CNN/belief rounds are part of each forward/backward pass; they do not multiply the number of independent samples. The joint version's CUDA memory use and throughput remain unvalidated and must not be inferred from the frozen version's measurements.

## Environment and execution

The validated environment uses Python 3.12, PyTorch 2.11 with CUDA 12.8, and Sionna 2.1. Dependencies are declared in each version's `bayesian_eqdeeprx/pyproject.toml`. Both versions use the same Python import package name: run from the selected version directory or use separate virtual environments.

The PowerShell scripts default to `D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe` and store runtime files under `D:\BayesianEqDeepRxRuns`. Override `-PythonPath` and `-RunDir` for another machine or environment.

### Frozen DenoiseNN version

```powershell
Set-Location .\width64_fast
# A new run: complete CUDA preflight in a separate directory, then train.
& .\bayesian_eqdeeprx\train.ps1 -Preflight -RunDir 'D:\BayesianEqDeepRxRuns\my_width64_run'
& .\bayesian_eqdeeprx\train.ps1 -RunDir 'D:\BayesianEqDeepRxRuns\my_width64_run'
# Resume an existing complete checkpoint:
& .\bayesian_eqdeeprx\train.ps1 -Resume -RunDir 'D:\BayesianEqDeepRxRuns\my_width64_run'
```

The included `pretrained_denoiser.pt` is a required fixed front-end asset. Generated model checkpoints, optimizer states, logs, and caches stay in the external run directory. Before launching, verify that the same training job is not already running.

### Full joint training version

```powershell
Set-Location .\width64_joint
# CPU smoke test with isolated outputs:
& .\bayesian_eqdeeprx\train.ps1 -Smoke
# Inspect the command without executing it:
& .\bayesian_eqdeeprx\train.ps1 -Train -DryRun
# When GPU resources are available, run preflight before formal training:
& .\bayesian_eqdeeprx\train.ps1 -Preflight
& .\bayesian_eqdeeprx\train.ps1 -Train
# Resume this version's complete checkpoint:
& .\bayesian_eqdeeprx\train.ps1 -Resume
```

The joint objective trains DenoiseNN, StateInit, all four StateUpdate networks, DemapperNN, and sigma using four-round bit CE and pre-projection soft-symbol losses. The frozen and joint versions use different checkpoint protocols and cannot directly resume each other's checkpoints. See the [joint version README](width64_joint/README.md) for tests and validation boundaries.

## Integrity and validation records

- Frozen source fingerprint: `6e540e10b99014ad718ba57a488e684ee122320a72d55d09641803cbc0310b89`.
- Joint source fingerprint: `df31da2cb5bfbb7161915c4fc25724b858fb137e5797da089e83a888f4becfe3`.
- Each `CODE_MANIFEST.json` records file hashes. The frozen version also includes `RUN_PROFILE.json` and `ACCEPTANCE.md`.
- References to a prepared 282-step checkpoint or training not yet started in the frozen delivery records are historical snapshots from October 24, 2026. Live progress is recorded only in the external run directory; this repository does not publish live training state.
- `.gitattributes` preserves original bytes and line endings so Git does not invalidate provenance hashes.

Only these two width-64 implementations and their supporting documentation are tracked. Papers, earlier large models, historical experiments, runtime checkpoints, logs, and caches remain local.
