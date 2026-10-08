# Download MorphoInstaller.exe from a GitHub release, install it silently on this
# Windows machine, and check that the installed morpho6 actually runs.
# The released binary looks up modules and help under C:/Program Files/Morpho.
# Failures are emitted as ::error:: annotations so they are visible without
# opening the raw Actions log.

$ErrorActionPreference = "Stop"
# A missing scheduled task is a diagnostic, not a terminating error.
$PSNativeCommandUseErrorActionPreference = $false

$Tag = if ($env:INSTALLER_TAG) { $env:INSTALLER_TAG } else { "v0.6.5-beta" }
$ExpectedRoot = "C:\Program Files\Morpho"
$ExpectedBin = Join-Path $ExpectedRoot "bin"
$ExpectedExe = Join-Path $ExpectedBin "morpho6.exe"
$RequiredNames = @(
    "morpho6.exe",
    "morpho.dll",
    "libopenblas.dll",
    "cxsparse.dll",
    "suitesparseconfig.dll"
)
$Failures = @()
$Problems = @()

# GitHub keeps only 10 notice annotations per step, so routine progress stays
# in the log and the step summary. Write-Result is reserved for results we
# need to read back from the public annotations API.
function Write-Note([string]$Message) {
    Write-Host $Message
    if ($env:GITHUB_STEP_SUMMARY) {
        Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value $Message
    }
}

function Write-Result([string]$Message) {
    Write-Note $Message
    Write-Host "::notice::$Message"
}

function Add-Failure([string]$Message) {
    Write-Host $Message
    Write-Host "::error::$Message"
    if ($env:GITHUB_STEP_SUMMARY) {
        Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value ("- " + $Message)
    }
    $script:Failures += $Message
}

# Reported as an error, but morpho6 can still be tested.
function Add-Problem([string]$Message) {
    Write-Host $Message
    Write-Host "::error::$Message"
    if ($env:GITHUB_STEP_SUMMARY) {
        Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value ("- " + $Message)
    }
    $script:Problems += $Message
}

function Find-Morpho6 {
    $Candidates = @(
        $ExpectedExe,
        "C:\Program Files (x86)\Morpho\bin\morpho6.exe",
        (Join-Path $env:LOCALAPPDATA "Morpho\bin\morpho6.exe"),
        (Join-Path $env:USERPROFILE "Morpho\bin\morpho6.exe"),
        (Join-Path $env:PUBLIC "Morpho\bin\morpho6.exe")
    )
    foreach ($Key in @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Morpho",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Morpho"
    )) {
        if (Test-Path -LiteralPath $Key) {
            $Loc = (Get-ItemProperty -LiteralPath $Key).InstallLocation
            if ($Loc) {
                $Candidates += (Join-Path $Loc.TrimEnd('\') "bin\morpho6.exe")
            }
        }
    }
    foreach ($Path in ($Candidates | Select-Object -Unique)) {
        if ($Path -and (Test-Path -LiteralPath $Path)) {
            return $Path
        }
    }
    return $null
}

Write-Note "Downloading MorphoInstaller.exe from release $Tag"
$Installer = Join-Path $env:RUNNER_TEMP "MorphoInstaller.exe"
$Url = "https://github.com/Morpho-lang/morpho/releases/download/$Tag/MorphoInstaller.exe"
Invoke-WebRequest -Uri $Url -OutFile $Installer
Unblock-File -Path $Installer
$Item = Get-Item $Installer
Write-Note ("Downloaded {0:N1} MB" -f ($Item.Length / 1MB))
if ($Item.Length -lt 1MB) {
    Add-Failure "Installer download is unexpectedly small ($($Item.Length) bytes)."
    throw ($Failures -join " ")
}

Write-Note "Starting silent install ($Installer /S)"
$Proc = Start-Process -FilePath $Installer -ArgumentList "/S" -PassThru
Write-Note "Installer pid $($Proc.Id)"

$Deadline = (Get-Date).AddMinutes(8)
$SawInstaller = $false
$IdleSince = $null
$ActualExe = $null
do {
    Start-Sleep -Seconds 2
    $Running = @(Get-Process -Name "MorphoInstaller" -ErrorAction SilentlyContinue)
    if ($Running.Count -gt 0) {
        $SawInstaller = $true
        $IdleSince = $null
    } elseif (-not $IdleSince) {
        $IdleSince = Get-Date
    }
    $ActualExe = Find-Morpho6
    if ($ActualExe -and $Running.Count -eq 0) { break }
    # The installer relaunches itself. Once that process is gone, stop waiting
    # even if morpho6 was not written to the expected directory.
    if ($SawInstaller -and $Running.Count -eq 0 -and $IdleSince -and
        ((Get-Date) - $IdleSince).TotalSeconds -ge 20) {
        break
    }
} while ((Get-Date) -lt $Deadline)

$ActualExe = Find-Morpho6
$StillRunning = @(Get-Process -Name "MorphoInstaller" -ErrorAction SilentlyContinue)
if ($StillRunning.Count -gt 0) {
    Write-Note "MorphoInstaller is still running after the wait (pids: $($StillRunning.Id -join ', '))."
}
if (-not $ActualExe) {
    Add-Failure "Silent install did not produce morpho6.exe. Expected $ExpectedExe. Installer process was seen: $SawInstaller."
    throw ($Failures -join " ")
}

$ActualBin = Split-Path -Parent $ActualExe
$ActualRoot = Split-Path -Parent $ActualBin
Write-Result "morpho6.exe installed at $ActualExe"
if ($ActualExe -ne $ExpectedExe) {
    Add-Failure "morpho6.exe is at '$ActualExe', but this build looks up modules and help in '$ExpectedRoot'."
}

$Missing = @()
foreach ($Name in $RequiredNames) {
    $Path = Join-Path $ActualBin $Name
    if (-not (Test-Path -LiteralPath $Path)) { $Missing += $Name }
}
if ($Missing.Count -gt 0) {
    Add-Failure ("Installed bin is missing: " + ($Missing -join ", "))
} else {
    Write-Note "Required DLLs are next to morpho6.exe."
}

$UninstallKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Morpho",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Morpho"
)
$Uninstall = $UninstallKeys | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Uninstall) {
    Add-Failure "Uninstall registry key Software\...\Uninstall\Morpho was not created."
} else {
    $Info = Get-ItemProperty -LiteralPath $Uninstall
    Write-Note "Uninstall key: $Uninstall"
    Write-Note "DisplayName=$($Info.DisplayName); DisplayVersion=$($Info.DisplayVersion); InstallLocation=$($Info.InstallLocation)"
    if ($Info.DisplayVersion -ne "0.6.5") {
        Add-Failure "Expected DisplayVersion 0.6.5, found '$($Info.DisplayVersion)'."
    }
}

function Format-PathHits([string]$Label, [string]$Value) {
    if (-not $Value) { return "$Label=(empty)" }
    $Hits = @($Value -split ';' | Where-Object { $_ -like '*Morpho*' })
    if ($Hits.Count -eq 0) { return "$Label=no Morpho entry" }
    return "$Label=[" + ($Hits -join " | ") + "]"
}

$MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$PathReport = @(
    (Format-PathHits "machine" $MachinePath),
    (Format-PathHits "user" $UserPath)
) -join "; "
Write-Result "PATH after install: $PathReport"
$OnPath = ($MachinePath -like '*\Morpho\bin*') -or ($UserPath -like '*\Morpho\bin*')
if (-not $OnPath) {
    Add-Problem "Installer did not add $ActualBin to the machine or user PATH ($PathReport)."
}
Add-Content -Path $env:GITHUB_PATH -Value $ActualBin

function Invoke-Morpho([string[]]$ArgumentList, [int]$TimeoutSec) {
    $OutFile = Join-Path $env:RUNNER_TEMP "morpho-out.txt"
    $ErrFile = Join-Path $env:RUNNER_TEMP "morpho-err.txt"
    Remove-Item -LiteralPath $OutFile, $ErrFile -ErrorAction SilentlyContinue
    $Child = Start-Process -FilePath $ActualExe -ArgumentList $ArgumentList -NoNewWindow -PassThru `
        -RedirectStandardOutput $OutFile -RedirectStandardError $ErrFile
    $Finished = $Child.WaitForExit($TimeoutSec * 1000)
    if (-not $Finished) {
        Stop-Process -Id $Child.Id -Force -ErrorAction SilentlyContinue
        return @{ Code = $null; Text = "timed out after $TimeoutSec seconds" }
    }
    $Text = @()
    if (Test-Path -LiteralPath $OutFile) { $Text += Get-Content -LiteralPath $OutFile }
    if (Test-Path -LiteralPath $ErrFile) { $Text += Get-Content -LiteralPath $ErrFile }
    return @{ Code = $Child.ExitCode; Text = ($Text -join "`n") }
}

$Version = Invoke-Morpho @("--version") 30
Write-Result "morpho6 --version exit=$($Version.Code) output=[$($Version.Text.Trim())]"
if ($Version.Code -ne 0 -or $Version.Text -notmatch "0\.6\.5") {
    Add-Failure "morpho6 --version did not report 0.6.5 (exit=$($Version.Code), output=$($Version.Text.Trim()))."
}

$Smoke = Join-Path $env:RUNNER_TEMP "smoke.morpho"
$Program = @"
print 1 + 1
import constants
print Pi
print System.platform()
print System.version()
"@
[System.IO.File]::WriteAllText($Smoke, $Program.Replace("`r`n", "`n"))
$SmokeResult = Invoke-Morpho @($Smoke) 60
$ActualLines = @($SmokeResult.Text -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
$ExpectedLines = @("2", "3.14159", "windows", "0.6.5")
Write-Result ("smoke exit=$($SmokeResult.Code) output=[" + ($ActualLines -join " | ") + "]")
$Same = ($SmokeResult.Code -eq 0) -and ($ActualLines.Count -eq $ExpectedLines.Count)
if ($Same) {
    for ($i = 0; $i -lt $ExpectedLines.Count; $i++) {
        if ($ActualLines[$i] -ne $ExpectedLines[$i]) { $Same = $false }
    }
}
if (-not $Same) {
    Add-Failure ("Smoke test expected '" + ($ExpectedLines -join " | ") + "' but got exit=$($SmokeResult.Code) output='" + ($ActualLines -join " | ") + "'.")
}

$PackageRoots = @(
    (Join-Path $env:USERPROFILE "morpho"),
    (Join-Path $env:PUBLIC "morpho"),
    "C:\Users\Public\morpho",
    $ActualRoot,
    $ExpectedRoot
) | Select-Object -Unique

$Done = $null
$PackageDeadline = (Get-Date).AddMinutes(2)
do {
    foreach ($Root in $PackageRoots) {
        $Candidate = Join-Path $Root "morpho-pkginstall.done"
        if (Test-Path -LiteralPath $Candidate) {
            $Done = $Candidate
            break
        }
    }
    if ($Done) { break }
    Start-Sleep -Seconds 5
} while ((Get-Date) -lt $PackageDeadline)

$Task = ((schtasks /Query /TN MorphoPackageInstall /FO LIST 2>&1 | Out-String) -replace '\s+', ' ').Trim()
if ($Task.Length -gt 300) { $Task = $Task.Substring(0, 300) }
$Listing = @()
foreach ($Root in $PackageRoots) {
    if (Test-Path -LiteralPath $Root) {
        $Names = @(Get-ChildItem -LiteralPath $Root -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
        $Listing += ("${Root}: " + ($Names -join ", "))
    }
}
$ListingText = ($Listing -join " || ")
if ($ListingText.Length -gt 500) { $ListingText = $ListingText.Substring(0, 500) }
if ($Done) {
    Write-Result "Package install marker: $Done"
} else {
    Add-Problem "Package install did not write morpho-pkginstall.done. Task: $Task. Dirs: $ListingText"
}

if ($Failures.Count -gt 0) {
    throw ($Failures -join " ")
}
if ($Problems.Count -gt 0) {
    # Keep the job red after the language tests, without skipping those tests.
    Add-Content -Path $env:GITHUB_ENV -Value "INSTALLER_PROBLEMS=1"
}
Write-Result "Installed morpho6 passed the smoke test."
