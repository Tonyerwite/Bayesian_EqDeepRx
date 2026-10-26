from types import SimpleNamespace
from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest

from bayesian_eqdeeprx import __main__ as cli


def test_cli_exposes_joint_smoke_and_no_retired_migration():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["migrate", "--run-dir", "D:/new", "--source-run-dir", "D:/old"])
    args = parser.parse_args(["smoke", "--run-dir", "D:/new"])
    assert args.cpu_threads == 1
    args = parser.parse_args(["train", "--run-dir", "D:/new", "--confirm"])
    assert args.steps == 70000 and args.seed == 2026


def test_cuda_admission_refuses_an_existing_gpu_process(monkeypatch):
    assert hasattr(cli, "require_exclusive_cuda"), "Need CUDA admission before allocation"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="5580\n", returncode=0))
    with pytest.raises(RuntimeError, match="GPU"):
        cli.require_exclusive_cuda()


def test_cpu_smoke_cannot_be_used_as_cuda_readiness():
    assert hasattr(cli, "validate_preflight"), "Need strict joint CUDA readiness validation"
    with pytest.raises(ValueError, match="preflight"):
        cli.validate_preflight({"long_training_ready": False, "cpu_smoke_passed": True},
                               {"long_training_ready": True, "training_mode": "full_joint"})


@pytest.mark.skipif(os.name != "nt", reason="Windows entry point")
def test_powershell_cpu_smoke_command_isolated_without_launching_training():
    shell = shutil.which("pwsh") or str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    script = Path(__file__).resolve().parents[1] / "bayesian_eqdeeprx/train.ps1"
    result = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                             "-Smoke", "-DryRun"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)
    assert command["arguments"][:3] == ["-m", "bayesian_eqdeeprx", "smoke"]
    assert command["cuda_visible_devices"] == "-1"
    assert "width64_joint_validation" in command["run_dir"]
    args = command["arguments"]
    assert args[args.index("--batch-size") + 1] == "1"


