param(
    [string]$BindHost = "0.0.0.0",
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

$previousHostedDemo = $env:REPIXELIZER_HOSTED_DEMO
$previousShowQueuePanel = $env:REPIXELIZER_SHOW_QUEUE_PANEL

try {
    $env:REPIXELIZER_HOSTED_DEMO = "1"

    if ($QueueUI -eq "show") {
        $env:REPIXELIZER_SHOW_QUEUE_PANEL = "1"
    } elseif ($QueueUI -eq "hide") {
        $env:REPIXELIZER_SHOW_QUEUE_PANEL = "0"
    } else {
        Remove-Item Env:\REPIXELIZER_SHOW_QUEUE_PANEL -ErrorAction SilentlyContinue
    }

    $launcherArgs = @(
        (Join-Path $repoRoot "scripts\run_gui.py"),
        "--host", $BindHost,
        "--port", $Port,
        "--queue-ui", $QueueUI
    )

    if ($Reload) {
        $launcherArgs += "--reload"
    }

    Write-Host "Starting Repixelizer hosted GUI..."
    Write-Host "  Hosted demo: $env:REPIXELIZER_HOSTED_DEMO"
    Write-Host "  Bind: http://${BindHost}:${Port}"
    Write-Host "  Python: $python"
    Write-Host ""

    & $python @launcherArgs
} finally {
    if ($null -eq $previousHostedDemo) {
        Remove-Item Env:\REPIXELIZER_HOSTED_DEMO -ErrorAction SilentlyContinue
    } else {
        $env:REPIXELIZER_HOSTED_DEMO = $previousHostedDemo
    }

    if ($null -eq $previousShowQueuePanel) {
        Remove-Item Env:\REPIXELIZER_SHOW_QUEUE_PANEL -ErrorAction SilentlyContinue
    } else {
        $env:REPIXELIZER_SHOW_QUEUE_PANEL = $previousShowQueuePanel
    }
}
