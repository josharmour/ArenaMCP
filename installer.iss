; Inno Setup Script for ArenaMCP
; Download Inno Setup from: https://jrsoftware.org/isinfo.php

#define MyAppName "ArenaMCP"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "ArenaMCP"
#define MyAppURL "https://github.com/yourusername/ArenaMCP"
#define MyAppExeName "ArenaMCP.exe"

[Setup]
AppId={{A7E8F4C3-2B1D-4E5F-9C0A-3B2D1E4F5C6A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=dist
OutputBaseFilename=ArenaMCP-{#MyAppVersion}-setup
SetupIconFile=assets\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "envfile"; Description: "Create .env configuration file"; GroupDescription: "Configuration:"; Flags: checkedonce

[Files]
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{#MyAppName} - Draft Helper"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--draft"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  EnvFile: string;
begin
  if CurStep = ssPostInstall then
  begin
    if IsTaskSelected('envfile') then
    begin
      EnvFile := ExpandConstant('{app}\.env');
      if not FileExists(EnvFile) then
      begin
        SaveStringToFile(EnvFile,
          '# ArenaMCP Configuration' + #13#10 +
          '# Add your API key below' + #13#10 +
          #13#10 +
          '# For Gemini (recommended)' + #13#10 +
          'GOOGLE_API_KEY=' + #13#10 +
          #13#10 +
          '# For Claude' + #13#10 +
          'ANTHROPIC_API_KEY=' + #13#10,
          False);
      end;
    end;
  end;
end;
