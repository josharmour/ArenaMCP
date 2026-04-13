param(
    [string]$InnoSetupCompiler = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)

$ErrorActionPreference = "Stop"

$ScriptPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
    $MyInvocation.MyCommand.Path
)
$ScriptDir = Split-Path -Parent $ScriptPath
$RepoRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
    (Join-Path $ScriptDir "..")
)
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

$LauncherProject = Join-Path $RepoRoot "installer\MtgaCoachLauncher\MtgaCoachLauncher.csproj"
if (-not (Test-Path $LauncherProject)) {
    throw "Launcher project not found: $LauncherProject"
}

$PublishRoot = Join-Path $env:TEMP "mtgacoach-installer-build"
$PublishDir = Join-Path $PublishRoot "launcher-publish"

Push-Location $ScriptDir
try {
    if (Test-Path $PublishRoot) {
        Remove-Item -Recurse -Force $PublishRoot
    }
    New-Item -ItemType Directory -Force -Path $PublishDir | Out-Null

    & dotnet publish $LauncherProject -c Release -p:Platform=x64 --self-contained `
        -p:BaseIntermediateOutputPath="$PublishRoot\obj\" `
        -p:BaseOutputPath="$PublishRoot\bin\" `
        -p:MSBuildProjectExtensionsPath="$PublishRoot\obj\" `
        -p:PublishDir="$PublishDir\"
    & $InnoSetupCompiler "/DAppVersion=$Version" "/DLauncherPublishDir=$PublishDir" "mtgacoach.iss"
}
finally {
    Pop-Location
}
