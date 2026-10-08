# Download MorphoInstaller.exe from a GitHub release, install it silently on this
# Windows machine, and check that the installed morpho6 actually runs.
# The released binary looks up modules and help under C:/Program Files/Morpho.

$ErrorActionPreference = "Stop"
# A missing scheduled task is a diagnostic, not a terminating error.
$PSNativeCommandUseErrorActionPreference = $false

$Tag = if ($env:INSTALLER_TAG) { $env:INSTALLER_TAG } else { "v0.6.5-beta" }
$InstallRoot = "C:\Program Files\Morpho"
$Bin = Join-Path $InstallRoot "bin"
$Morpho6 = Join-Path $Bin "morpho6.exe"
$Required = @(
    "morpho6.exe",
    "morpho.dll",
    "libopenblas.dll",
    "cxsparse.dll",
    "suitesparseconfig.dll"
)

function Write-Step([string]$Message) {
    Write-Host "::group::$Message"
}

function End-Step {
    Write-Host "::endgroup::"
}

Write-Step "Download MorphoInstaller.exe from $Tag"
$Installer = Join-Path $env:RUNNER_TEMP "MorphoInstaller.exe"
$Url = "https://github.com/Morpho-lang/morpho/releases/download/$Tag/MorphoInstaller.exe"
Write-Host "GET $Url"
Invoke-WebRequest -Uri $Url -OutFile $Installer
Unblock-File -Path $Installer
$Item = Get-Item $Installer
Write-Host ("Downloaded {0:N1} MB" -f ($Item.Length / 1MB))
if ($Item.Length -lt 1MB) {
    throw "Installer download is unexpectedly small ($($Item.Length) bytes)."
}
End-Step

Write-Step "Silent install"
# NSIS relaunches itself, so the first process can exit before setup finishes.
$Proc = Start-Process -FilePath $Installer -ArgumentList "/S" -PassThru
Write-Host "Started installer pid $($Proc.Id)"
$Deadline = (Get-Date).AddMinutes(10)
do {
    Start-Sleep -Seconds 2
    $Running = @(Get-Process -Name "MorphoInstaller" -ErrorAction SilentlyContinue)
    if ((Test-Path -LiteralPath $Morpho6) -and $Running.Count -eq 0) {
        break
    }
} while ((Get-Date) -lt $Deadline)

if (-not (Test-Path -LiteralPath $Morpho6)) {
    Write-Host "morpho6.exe was not installed to $Morpho6"
    foreach ($Root in @("C:\Program Files", "C:\Program Files (x86)", $env:LOCALAPPDATA, $env:USERPROFILE)) {
        if (-not (Test-Path -LiteralPath $Root)) { continue }
        Get-ChildItem -Path $Root -Filter "morpho6.exe" -Recurse -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName |
            ForEach-Object { Write-Host "found $_" }
    }
    throw "Silent install did not produce $Morpho6. The released binary expects this path."
}

foreach ($Name in $Required) {
    $Path = Join-Path $Bin $Name
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Installed tree is missing $Path"
    }
}
Write-Host "Installed files:"
Get-ChildItem -LiteralPath $InstallRoot | ForEach-Object { Write-Host ("  " + $_.Name) }
End-Step

Write-Step "Installer registration"
$UninstallKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Morpho",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Morpho"
)
$Uninstall = $UninstallKeys | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Uninstall) {
    throw "Uninstall registry key Software\...\Uninstall\Morpho was not created."
}
$Info = Get-ItemProperty -LiteralPath $Uninstall
Write-Host "DisplayName: $($Info.DisplayName)"
Write-Host "DisplayVersion: $($Info.DisplayVersion)"
Write-Host "InstallLocation: $($Info.InstallLocation)"
Write-Host "UninstallString: $($Info.UninstallString)"
if ($Info.DisplayVersion -ne "0.6.5") {
    throw "Expected DisplayVersion 0.6.5, found '$($Info.DisplayVersion)'."
}

$MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
Write-Host "Machine PATH contains Morpho bin: $($MachinePath -like '*\Morpho\bin*')"
Write-Host "User PATH contains Morpho bin: $($UserPath -like '*\Morpho\bin*')"
if (($MachinePath -notlike '*\Morpho\bin*') -and ($UserPath -notlike '*\Morpho\bin*')) {
    throw "Installer did not add $Bin to the machine or user PATH."
}
Add-Content -Path $env:GITHUB_PATH -Value $Bin
End-Step

Write-Step "Bundled package install"
# The installer drops package archives and finishes them with a scheduled task.
$PackageRoots = @(
    (Join-Path $env:USERPROFILE "morpho"),
    (Join-Path $env:PUBLIC "morpho"),
    "C:\Users\Public\morpho",
    $InstallRoot
) | Select-Object -Unique

$Done = $null
$PackageDeadline = (Get-Date).AddMinutes(3)
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

Write-Host "schtasks MorphoPackageInstall:"
schtasks /Query /TN MorphoPackageInstall /V /FO LIST 2>&1 | ForEach-Object { Write-Host $_ }

foreach ($Root in $PackageRoots) {
    if (Test-Path -LiteralPath $Root) {
        Write-Host "Contents of $Root :"
        Get-ChildItem -LiteralPath $Root -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host ("  " + $_.Name) }
    }
}

$Bat = $PackageRoots |
    ForEach-Object { Join-Path $_ "morpho-install-packages.bat" } |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if ($Bat) {
    Write-Host "---- $Bat ----"
    Get-Content -LiteralPath $Bat | ForEach-Object { Write-Host $_ }
}

if (-not $Done) {
    throw "Package install did not write morpho-pkginstall.done within 3 minutes."
}
Write-Host "Package install finished: $Done"
End-Step

Write-Step "Run installed morpho6"
$Smoke = Join-Path $env:RUNNER_TEMP "smoke.morpho"
$Program = @"
print 1 + 1
import constants
print Pi
print System.platform()
print System.version()
"@
[System.IO.File]::WriteAllText($Smoke, $Program.Replace("`r`n", "`n"))

$Output = & $Morpho6 $Smoke 2>&1 | ForEach-Object { "$_" }
$Code = $LASTEXITCODE
Write-Host "exit code: $Code"
$Output | ForEach-Object { Write-Host $_ }

$Expected = @("2", "3.14159", "windows", "0.6.5")
$Actual = @($Output | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
$Same = ($Actual.Count -eq $Expected.Count)
if ($Same) {
    for ($i = 0; $i -lt $Expected.Count; $i++) {
        if ($Actual[$i] -ne $Expected[$i]) { $Same = $false }
    }
}
if ((-not $Same) -or ($Code -ne 0)) {
    Write-Host "Expected:"
    $Expected | ForEach-Object { Write-Host "  $_" }
    throw "Installed morpho6 did not produce the expected smoke-test output."
}
Write-Host "Smoke test passed."
End-Step
