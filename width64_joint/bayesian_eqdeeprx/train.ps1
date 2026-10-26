[CmdletBinding()]
param(
    [switch]$Smoke,
    [switch]$Preflight,
    [switch]$Train,
    [switch]$Resume,
    [switch]$DryRun,
    [string]$RunDir = '',
    [int]$BatchSize = 112,
    [int]$GenerationBatchSize = 2,
    [int]$ModelMicrobatchSize = 2,
    [int]$MessageChunkSize = 8192,
    [int]$CpuThreads = 4,
    [int]$Repeats = 2,
    [int]$Seed = 2026,
    [string]$PythonPath = 'D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$selectedModes = @($Smoke, $Preflight, $Train, $Resume | Where-Object { $_ })
if ($selectedModes.Count -ne 1) {
    throw 'Select exactly one operation: -Smoke, -Preflight, -Train, or -Resume.'
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python was not found: $PythonPath"
}
$workspace = Split-Path -Parent $PSScriptRoot
if ($Smoke) {
    if (-not $RunDir) { $RunDir = 'D:\BayesianEqDeepRxRuns\width64_joint_validation\cpu_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
    if (-not $PSBoundParameters.ContainsKey('BatchSize')) { $BatchSize = 1 }
    if (-not $PSBoundParameters.ContainsKey('GenerationBatchSize')) { $GenerationBatchSize = 1 }
    if (-not $PSBoundParameters.ContainsKey('ModelMicrobatchSize')) { $ModelMicrobatchSize = 1 }
    if (-not $PSBoundParameters.ContainsKey('MessageChunkSize')) { $MessageChunkSize = 128 }
    if (-not $PSBoundParameters.ContainsKey('CpuThreads')) { $CpuThreads = 1 }
    if (-not $PSBoundParameters.ContainsKey('Repeats')) { $Repeats = 1 }
} elseif (-not $RunDir) {
    $RunDir = 'D:\BayesianEqDeepRxRuns\width64_joint_batch112'
}
$RunDir = [System.IO.Path]::GetFullPath($RunDir).TrimEnd('\')
$project = Split-Path -Parent $workspace
$protectedPaths = @(
    'D:\BayesianEqDeepRxRuns\width64_fast_batch112',
    'D:\BayesianEqDeepRxRuns\width64_batch112_match46776',
    'D:\BayesianEqDeepRxRuns\batch8_70000',
    'D:\BayesianEqDeepRxRuns\strict_70000',
    'D:\EqDeepRxRuns',
    (Join-Path $project 'width64_fast'),
    (Join-Path $project 'bayesian_eqdeeprx'),
    'C:\Users\Chen Yan\Desktop\eqdeeprx最终复现'
)
foreach ($protectedPath in $protectedPaths) {
    $fullProtected = [System.IO.Path]::GetFullPath($protectedPath).TrimEnd('\')
    if ($RunDir.Equals($fullProtected, [StringComparison]::OrdinalIgnoreCase) -or
        $RunDir.StartsWith($fullProtected + '\', [StringComparison]::OrdinalIgnoreCase) -or
        $fullProtected.StartsWith($RunDir + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "protected existing run/source directory: $RunDir"
    }
}
$operation = if ($Smoke) { 'smoke' } elseif ($Preflight) { 'preflight' } else { 'train' }
$arguments = @('-m', 'bayesian_eqdeeprx', $operation, '--run-dir', $RunDir,
    '--batch-size', "$BatchSize", '--generation-batch-size', "$GenerationBatchSize",
    '--model-microbatch-size', "$ModelMicrobatchSize", '--message-chunk-size', "$MessageChunkSize",
    '--cpu-threads', "$CpuThreads")
if ($Smoke -or $Preflight) {
    $arguments += @('--repeats', "$Repeats")
} else {
    $arguments += @('--steps', '70000', '--save-every', '1', '--seed', "$Seed", '--confirm')
    if ($Resume) { $arguments += '--resume' }
}
if (-not $Smoke) { $arguments += @('--factor-gather', '--geometry-graphs') }
if ($DryRun) {
    [ordered]@{ python=$PythonPath; working_directory=$workspace; run_dir=$RunDir;
        cuda_visible_devices=$(if ($Smoke) { '-1' } else { $env:CUDA_VISIBLE_DEVICES });
        arguments=$arguments } | ConvertTo-Json -Depth 4
    exit 0
}
$runtime = Join-Path $RunDir 'runtime'
foreach ($name in @('temp', 'torch', 'xdg', 'triton', 'torchinductor', 'cuda', 'mpl', 'pycache')) {
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
$env:PYTHONPYCACHEPREFIX = Join-Path $runtime 'pycache'
$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
if ($Smoke) { $env:CUDA_VISIBLE_DEVICES = '-1' }
$exitCode = 1
Push-Location -LiteralPath $workspace
try {
    $priorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonPath @arguments
        $exitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $priorPreference }
} finally { Pop-Location }
exit $exitCode
