; ============================================================
; Metabobarcoding Installer Script
; Inno Setup 6.x
;
; To build:
;   1. Install Inno Setup from https://jrsoftware.org/isdl.php
;   2. Run build_bundle.bat first to create dist\Metabobarcoding\
;   3. Right-click this file → Compile  (or run: iscc installer.iss)
;   4. Output: dist\Metabobarcoding_Setup.exe
; ============================================================

#define AppName      "Metabobarcoding"
#define AppVersion   "1.0"
#define AppPublisher "Coskun Lab - Georgia Tech"
#define AppURL       "https://github.com/coskunlab/Metabobarcoding"
#define BundleDir    "..\dist\Metabobarcoding"
#define OutputDir    "..\dist"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir={#OutputDir}
OutputBaseFilename=Metabobarcoding_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Require at least 20 GB free disk space
ExtraDiskSpaceRequired=21474836480
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Bundle all files from dist\Metabobarcoding\
; Excludes the .tar.gz archives after first-run unpacking is done
; (they are still included so first-run setup works)
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start menu shortcut
Name: "{group}\{#AppName}"; Filename: "{app}\launch.bat"; \
    WorkingDir: "{app}"; \
    Flags: runminimized

; Desktop shortcut (optional)
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\launch.bat"; \
    WorkingDir: "{app}"; \
    Flags: runminimized; Tasks: desktopicon

; Uninstall shortcut in start menu
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[Run]
; Offer to launch after install
Filename: "{app}\launch.bat"; Description: "{cm:LaunchProgram,{#AppName}}"; \
    Flags: nowait postinstall skipifsilent runminimized

[UninstallDelete]
; Clean up unpacked envs on uninstall
Type: filesandordirs; Name: "{app}\envs\torch_gpu3"
Type: filesandordirs; Name: "{app}\envs\mesmer"
Type: filesandordirs; Name: "{app}\deepcell_home"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nThe installation requires approximately 15 GB of disk space.%n%nThe first time you launch the app, it will unpack the Python environments (5-10 minutes). This only happens once.%n%nClick Next to continue.
