$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "`n[1/3] Creating venv (inherits system torch/transformers/datasets)..." -ForegroundColor Cyan
python -m venv "$root\.venv-mi" --system-site-packages

$pip = "$root\.venv-mi\Scripts\pip.exe"
$py  = "$root\.venv-mi\Scripts\python.exe"

Write-Host "[2/3] Installing RelP TL fork + missing deps..." -ForegroundColor Cyan
& $pip install -e "$root\RelP\TransformerLens" --no-deps -q
& $pip install `
    "einops>=0.6.0" `
    "fancy-einsum>=0.0.3" `
    "jaxtyping>=0.2.11" `
    "beartype>=0.14.1,<0.15.0" `
    "better-abc>=0.0.3" `
    "rich>=12.6.0" `
    "tqdm>=4.64.1" `
    "wandb>=0.13.5" `
    "typeguard>=4.2" `
    "circuitsvis" `
    "scikit-learn" `
    "pandas" -q

Write-Host "[3/3] Running environment verification..." -ForegroundColor Cyan
& $py "$root\stage0_environment\verify_env.py"
