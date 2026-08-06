# setup_task_scheduler.ps1
# -------------------------
# Registers a Windows Task Scheduler job that runs the traffic collector
# every 20 minutes, indefinitely.
#
# Run this script ONCE as Administrator:
#   Right-click PowerShell -> "Run as Administrator"
#   cd "C:\Users\_arun_37_\OneDrive\Documents\GitHub\JustSteer"
#   .\setup_task_scheduler.ps1
#
# To check the task afterwards:
#   Get-ScheduledTask -TaskName "JustSteer_TrafficCollector"
#
# To remove the task:
#   Unregister-ScheduledTask -TaskName "JustSteer_TrafficCollector" -Confirm:$false

# ── Paths — update these if your project is in a different location ────────────
$ProjectRoot = "C:\Users\_arun_37_\OneDrive\Documents\GitHub\JustSteer"
$PythonExe   = "$ProjectRoot\mlops\Scripts\python.exe"
$Script      = "-m src.traffic_collector.collector"
$TaskName    = "JustSteer_TrafficCollector"
$LogFile     = "$ProjectRoot\logs\task_scheduler.log"

# ── Validate paths before registering ─────────────────────────────────────────
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at: $PythonExe"
    Write-Error "Update the `$PythonExe variable in this script."
    exit 1
}

if (-not (Test-Path $ProjectRoot)) {
    Write-Error "Project root not found at: $ProjectRoot"
    exit 1
}

# Create logs directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "$ProjectRoot\logs" | Out-Null

Write-Host ""
Write-Host "Setting up Task Scheduler job..." -ForegroundColor Cyan
Write-Host "  Task name   : $TaskName"
Write-Host "  Python      : $PythonExe"
Write-Host "  Working dir : $ProjectRoot"
Write-Host "  Interval    : Every 20 minutes"
Write-Host "  Log file    : $LogFile"
Write-Host ""

# ── Remove existing task if it exists ─────────────────────────────────────────
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task '$TaskName'..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# ── Build the action ───────────────────────────────────────────────────────────
# The action runs: python.exe -m src.traffic_collector.collector
# stdout and stderr are redirected to the log file via cmd /c
$CmdArgs = "/c `"cd /d `"$ProjectRoot`" && `"$PythonExe`" $Script >> `"$LogFile`" 2>&1`""

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument $CmdArgs `
    -WorkingDirectory $ProjectRoot

# ── Build the trigger — every 20 minutes, starting now, no end date ───────────
$StartTime = (Get-Date).AddMinutes(1)   # start 1 minute from now

$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $StartTime `
    -RepetitionInterval  (New-TimeSpan -Minutes 20) `
    -RepetitionDuration  (New-TimeSpan -Days 9999)

# ── Settings — run even when on battery, don't stop if running long ───────────
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit  (New-TimeSpan -Minutes 5) `
    -RestartCount        3 `
    -RestartInterval     (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -RunOnlyIfNetworkAvailable

# ── Principal — run as current user ───────────────────────────────────────────
$Principal = New-ScheduledTaskPrincipal `
    -UserId    ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel  Highest

# ── Register ───────────────────────────────────────────────────────────────────
Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $Action `
    -Trigger   $Trigger `
    -Settings  $Settings `
    -Principal $Principal `
    -Force | Out-Null

# ── Verify ────────────────────────────────────────────────────────────────────
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($task) {
    Write-Host "Task registered successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "Task details:"
    Write-Host "  State        : $($task.State)"
    Write-Host "  Next run     : $(($task | Get-ScheduledTaskInfo).NextRunTime)"
    Write-Host ""
    Write-Host "Useful commands:" -ForegroundColor Cyan
    Write-Host "  # Run manually right now:"
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    Write-Host "  # Check last run result:"
    Write-Host "  (Get-ScheduledTaskInfo -TaskName '$TaskName').LastTaskResult"
    Write-Host ""
    Write-Host "  # View live log:"
    Write-Host "  Get-Content '$LogFile' -Wait -Tail 20"
    Write-Host ""
    Write-Host "  # Stop the task:"
    Write-Host "  Stop-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    Write-Host "  # Remove the task permanently:"
    Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} else {
    Write-Error "Task registration failed. Run PowerShell as Administrator and try again."
    exit 1
}
