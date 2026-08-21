<#
.SYNOPSIS
    Pulls the latest DB snapshots from the GitHub Actions data-sync branch
    and merges them into your local SQLite databases without losing any rows.

.DESCRIPTION
    The daily_data_collection.yml cron job collects traffic and weather data
    on GitHub Actions and pushes both SQLite databases to a dedicated
    data-sync branch after every run.

    This script:
      1. Adds a temporary git worktree for the data-sync branch
      2. Runs scripts/merge_artifact_db.py to merge the remote DBs into
         your local ones (INSERT OR IGNORE - zero data loss)
      3. Removes the worktree

    Run this whenever you want to pull the latest cloud-collected data.

.PARAMETER DryRun
    Preview what would be merged without writing any changes.

.PARAMETER Python
    Path to the Python interpreter to use.
    Defaults to "python" (whatever is on PATH).

.EXAMPLE
    # Normal sync
    .\scripts\sync_local_dbs.ps1

.EXAMPLE
    # Preview only - no writes
    .\scripts\sync_local_dbs.ps1 -DryRun

.EXAMPLE
    # Use a specific venv
    .\scripts\sync_local_dbs.ps1 -Python ".\mlops\Scripts\python.exe"
#>

[CmdletBinding()]
param(
    [switch] $DryRun,
    [string] $Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Resolve repo root ────────────────────────────────────────────────────────
$RepoRoot   = (Resolve-Path "$PSScriptRoot\..").Path
$MergeScript = Join-Path $RepoRoot "scripts\merge_artifact_db.py"
$WorktreeDir = Join-Path $RepoRoot ".data-sync-worktree"

# ── Helpers ──────────────────────────────────────────────────────────────────
function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Write-OK([string]$msg) {
    Write-Host "    [OK] $msg" -ForegroundColor Green
}

function Write-Warn([string]$msg) {
    Write-Host "    [!]  $msg" -ForegroundColor Yellow
}

function Remove-Worktree {
    if (Test-Path $WorktreeDir) {
        Write-Step "Cleaning up worktree..."
        Push-Location $RepoRoot
        try {
            git worktree remove --force $WorktreeDir 2>$null
        } catch {
            # Fallback: just delete the directory if git worktree remove fails
            Remove-Item -Recurse -Force $WorktreeDir -ErrorAction SilentlyContinue
        }
        Pop-Location
        Write-OK "Worktree removed."
    }
}

# ── Pre-flight checks ────────────────────────────────────────────────────────
Write-Step "Pre-flight checks"

if (-not (Test-Path $MergeScript)) {
    Write-Error "merge_artifact_db.py not found at: $MergeScript"
    exit 1
}

# Ensure we are inside the repo root
Push-Location $RepoRoot
try {
    $null = git rev-parse --show-toplevel 2>&1
} catch {
    Write-Error "Not inside a git repository. Run this script from within the JustSteer repo."
    exit 1
}

Write-OK "Repo root: $RepoRoot"

# ── Fetch the data-sync branch ───────────────────────────────────────────────
Write-Step "Fetching data-sync branch from origin..."

cmd /c "git fetch origin data-sync 2>&1" | ForEach-Object { Write-Host "    $_" }

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to fetch origin/data-sync. Has the cron job run at least once?"
    Pop-Location
    exit 1
}

Write-OK "Fetched origin/data-sync"

# ── Check what's on the remote ───────────────────────────────────────────────
Write-Step "Checking remote DB timestamps..."

# Show last commit message on data-sync so the user knows how fresh it is
$LastCommit = git log origin/data-sync -1 --format="%s  (%cr)" 2>$null
if ($LastCommit) {
    Write-Host "    Last sync commit: $LastCommit" -ForegroundColor White
}

# ── Clean up any leftover worktree from a previous interrupted run ────────────
if (Test-Path $WorktreeDir) {
    Write-Warn "Leftover worktree found at $WorktreeDir - removing it first."
    git worktree remove --force $WorktreeDir 2>$null
    if (Test-Path $WorktreeDir) {
        Remove-Item -Recurse -Force $WorktreeDir
    }
}

# ── Add a temporary worktree for data-sync ───────────────────────────────────
Write-Step "Checking out data-sync branch into temporary worktree..."

cmd /c "git worktree add `"$WorktreeDir`" origin/data-sync 2>&1" | ForEach-Object { Write-Host "    $_" }

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create worktree. See output above."
    Pop-Location
    exit 1
}

Write-OK "Worktree ready at: $WorktreeDir"

# ── Verify DB files exist in the worktree ────────────────────────────────────
$RemoteTrafficDB  = Join-Path $WorktreeDir "data\traffic_history.db"
$RemoteFeaturesDB = Join-Path $WorktreeDir "data\db\daily_features.db"

$hasTraffic  = Test-Path $RemoteTrafficDB
$hasFeatures = Test-Path $RemoteFeaturesDB

if (-not $hasTraffic -and -not $hasFeatures) {
    Write-Warn "Neither DB file found in data-sync branch. Nothing to merge."
    Remove-Worktree
    Pop-Location
    exit 0
}

if ($hasTraffic)  { Write-OK "Found: data\traffic_history.db" }
else              { Write-Warn "data\traffic_history.db not in data-sync (will be skipped)" }

if ($hasFeatures) { Write-OK "Found: data\db\daily_features.db" }
else              { Write-Warn "data\db\daily_features.db not in data-sync (will be skipped)" }

# ── Build merge_artifact_db.py arguments ─────────────────────────────────────
$MergeArgs = @(
    $MergeScript,
    "--artifact-dir", $WorktreeDir
)

if ($DryRun) {
    $MergeArgs += "--dry-run"
    Write-Step "DRY RUN - previewing merge (no writes)..."
} else {
    Write-Step "Merging remote DBs into local databases..."
}

# ── Run the merge ─────────────────────────────────────────────────────────────
& $Python @MergeArgs

$MergeExitCode = $LASTEXITCODE

# ── Cleanup ───────────────────────────────────────────────────────────────────
Remove-Worktree
Pop-Location

# ── Final status ──────────────────────────────────────────────────────────────
if ($MergeExitCode -eq 0) {
    if ($DryRun) {
        Write-Host "`nDry run complete - no data was changed." -ForegroundColor Cyan
    } else {
        Write-Host "`nSync complete. Your local DBs are up to date." -ForegroundColor Green
        Write-Host "    Run: sqlite3 data/traffic_history.db `"SELECT MAX(sampled_at_ist) FROM traffic_snapshots;`"" -ForegroundColor DarkGray
        Write-Host "    to confirm the newest record." -ForegroundColor DarkGray
    }
} else {
    Write-Host "`nMerge script exited with code $MergeExitCode - check output above." -ForegroundColor Red
    exit $MergeExitCode
}
