param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload,
    [ValidateSet("auto", "show", "hide")]
    [string]$QueueUI = "auto"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$args = @(
    (Join-Path $repoRoot "scripts\run_gui.py"),
    "--host", $BindHost,
    "--port", $Port,
    "--queue-ui", $QueueUI
)

if ($Reload) {
    $args += "--reload"
}

& $python @args
