<#
.SYNOPSIS
    Run one char_lstm training to completion, outside the session's job object.

.DESCRIPTION
    A Claude Code background shell inherits a Windows job object from the
    session, so every process it spawns is terminated when the session ends
    or is interrupted. A training run is minutes-to-hours long and therefore
    outlives the thing holding its leash. On 2026-09-03 the Kazakh v6 run
    died twice that way, both times mid-epoch with no traceback and no
    non-zero exit: the tell that it was the job boundary rather than a crash.

    Task Scheduler places the job under services.exe instead, where nothing
    the session does can reach it. The trainer writes a .resume file every
    epoch, so a run interrupted this way restarts at the next epoch rather
    than from zero.

    Two registration traps, both measured previously and both handled here:

    Priority. The Register-ScheduledTask default is Priority 7, which is
    below-normal CPU AND low I/O AND background memory priority, inherited by
    every child. A previous run crawled at 0.16 cores on an idle 24-core box
    under it. This registers Priority 4.

    Principal. USERDOMAIN is WORKGROUP on this machine, and passing
    "WORKGROUP\test" as UserId fails registration with 0x80070534. The SID is
    unambiguous, so that is what is passed.

.PARAMETER Language
    Language code to train, one of the codes in char_lstm.corpora.LANGS.

.PARAMETER CorpusDir
    Directory holding the cleaned corpora, relative to the repository root.

.PARAMETER CheckpointDir
    Directory for checkpoints and the resume state, relative to the root.

.PARAMETER LogFile
    File the run appends its output to, relative to the repository root.

.EXAMPLE
    .\scripts\run-training-detached.ps1 -Language kk `
        -CorpusDir rebuild_2026-09\corpora_clean_v6 `
        -CheckpointDir checkpoints_v6 -LogFile train_v6.log
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Language,
    [Parameter(Mandatory = $true)][string]$CorpusDir,
    [Parameter(Mandatory = $true)][string]$CheckpointDir,
    [Parameter(Mandatory = $true)][string]$LogFile
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "No interpreter at $python; run 'make install' in $repo first."
}

$taskName = "char-lstm-train-$Language"

# The trainer is invoked through cmd so that append-redirection of both
# streams is the shell's job rather than something this script has to model.
$command = '"{0}" -m char_lstm.train --lang {1} --device cuda --corpus-dir "{2}" --checkpoint-dir "{3}"' -f `
    $python, $Language, $CorpusDir, $CheckpointDir
$arguments = '/c cd /d "{0}" && {1} >> "{2}" 2>&1' -f $repo, $command, $LogFile

$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\cmd.exe" -Argument $arguments

# S4U runs whether or not the user is logged on and stores no password. The
# SID rather than a domain-qualified name, per the registration trap above.
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$principal = New-ScheduledTaskPrincipal -UserId $sid -LogonType S4U -RunLevel Limited

# Zero time limit because the default terminates a task after three days,
# and IgnoreNew so a second start cannot put two writers on one checkpoint.
$settings = New-ScheduledTaskSettingsSet `
    -Priority 4 `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal `
    -Settings $settings -Description "char_lstm training for $Language" -Force | Out-Null

Start-ScheduledTask -TaskName $taskName

Write-Output "registered and started: $taskName"
Write-Output "  language   : $Language"
Write-Output "  corpus     : $CorpusDir"
Write-Output "  checkpoints: $CheckpointDir"
Write-Output "  log        : $LogFile"
Write-Output ""
Write-Output "state:   Get-ScheduledTask -TaskName $taskName | Select-Object State"
Write-Output "stop:    Stop-ScheduledTask -TaskName $taskName"
Write-Output "remove:  Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false"
