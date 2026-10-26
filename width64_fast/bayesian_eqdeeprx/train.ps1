[CmdletBinding()]
param(
    [switch]$Preflight,
    [switch]$Migrate,
    [switch]$Resume,
    [string]$RunDir = 'D:\BayesianEqDeepRxRuns\width64_fast_batch112',
    [string]$SourceRunDir = 'D:\BayesianEqDeepRxRuns\width64_batch112_match46776',
    [int]$BatchSize = 112,
    [int]$Steps = 70000,
    [int]$GenerationBatchSize = 2,
    [int]$MessageChunkSize = 8192,
    [int]$ModelMicrobatchSize = 8,
    [int]$Repeats = 2,
    [int]$Seed = 1002026,
    [string]$PythonPath = 'D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "CUDA Python was not found: $PythonPath"
}
$workspace = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$runtime = Join-Path $RunDir 'runtime'
foreach ($name in @('temp', 'torch', 'xdg', 'triton', 'torchinductor', 'cuda', 'mpl')) {
    New-Item -ItemType Directory -Path (Join-Path $runtime $name) -Force | Out-Null
}
$env:PYTORCH_ALLOC_CONF = 'max_split_size_mb:256'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TEMP = Join-Path $runtime 'temp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:TORCH_HOME = Join-Path $runtime 'torch'
$env:XDG_CACHE_HOME = Join-Path $runtime 'xdg'
$env:TRITON_CACHE_DIR = Join-Path $runtime 'triton'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $runtime 'torchinductor'
$env:CUDA_CACHE_PATH = Join-Path $runtime 'cuda'
$env:MPLCONFIGDIR = Join-Path $runtime 'mpl'
$arguments = if ($Preflight) {
    @('-m', 'bayesian_eqdeeprx', 'preflight', '--run-dir', $RunDir,
      '--batch-size', "$BatchSize", '--generation-batch-size', "$GenerationBatchSize",
      '--message-chunk-size', "$MessageChunkSize", '--model-microbatch-size', "$ModelMicrobatchSize",
      '--factor-gather', '--geometry-graphs', '--repeats', "$Repeats")
} elseif ($Migrate) {
    @('-m', 'bayesian_eqdeeprx', 'migrate', '--source-run-dir', $SourceRunDir,
      '--run-dir', $RunDir, '--batch-size', "$BatchSize",
      '--generation-batch-size', "$GenerationBatchSize", '--seed', "$Seed")
} else {
    @('-m', 'bayesian_eqdeeprx', 'train', '--run-dir', $RunDir,
      '--batch-size', "$BatchSize", '--seed', "$Seed",
      '--steps', "$Steps", '--save-every', '1', '--confirm')
}
if ($Resume -and -not $Preflight) { $arguments += '--resume' }
$exitCode = 1
Push-Location -LiteralPath $workspace
try {
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonPath @arguments
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $priorErrorActionPreference }
}
finally { Pop-Location }
exit $exitCode
