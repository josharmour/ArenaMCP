param(
    [string]$InnoSetupCompiler = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$Pyproject = Join-Path $RepoRoot "pyproject.toml"

if (-not (Test-Path $InnoSetupCompiler)) {
    throw "Inno Setup compiler not found: $InnoSetupCompiler"
}

$VersionLine = Select-String -Path $Pyproject -Pattern '^\s*version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $VersionLine) {
    throw "Could not read version from $Pyproject"
}

$Version = $VersionLine.Matches[0].Groups[1].Value
Write-Host "Building mtgacoach installer for v$Version"

$PluginDll = Join-Path $RepoRoot "bepinex-plugin\MtgaCoachBridge\bin\Release\net472\MtgaCoachBridge.dll"
if (-not (Test-Path $PluginDll)) {
    Write-Warning "Bridge plugin DLL not found at $PluginDll"
    Write-Warning "Build the plugin before cutting a release installer."
}

$ReleaseBuilder = Join-Path $RepoRoot "scripts\build_release.ps1"
if (-not (Test-Path $ReleaseBuilder)) {
    throw "Desktop release builder not found: $ReleaseBuilder"
}

Push-Location $ScriptDir
try {
    & $ReleaseBuilder
    & $InnoSetupCompiler "/DAppVersion=$Version" "mtgacoach.iss"
}
finally {
    Pop-Location
}
