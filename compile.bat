@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   🚀 Projet Gaia - Compilation du Moteur Natif Rust (Windows)
echo ============================================================

rem Verification de cargo
where cargo >nul 2>nul
if %ERRORLEVEL% neq 0 (
    if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
        set "PATH=%USERPROFILE%\.cargo\bin;!PATH!"
    ) else (
        echo [ERREUR] Le compilateur Rust 'cargo' n'a pas ete trouve dans le PATH.
        echo Installez Rust depuis https://rustup.rs/ puis reessayez.
        exit /b 1
    )
)

rem Verification de MinGW / dlltool (w64devkit)
where dlltool >nul 2>nul
if %ERRORLEVEL% neq 0 (
    if exist "%USERPROFILE%\w64devkit\bin\dlltool.exe" (
        set "PATH=%USERPROFILE%\w64devkit\bin;!PATH!"
    )
)

rem Definition d'un dossier cible ASCII pour eviter les erreurs d'accentuation MinGW/OneDrive
if "%CARGO_TARGET_DIR%"=="" (
    if exist "%USERPROFILE%\cargo_target" (
        set "TARGET_DIR=%USERPROFILE%\cargo_target"
    ) else (
        set "TARGET_DIR=%LOCALAPPDATA%\cargo_gaiapi_target"
    )
) else (
    set "TARGET_DIR=%CARGO_TARGET_DIR%"
)

cd /d "%~dp0projet_gaiapi"
echo [*] Compilation de projet_gaiapi en mode release (C-ABI)...
echo [*] Dossier cible de build : !TARGET_DIR!
cargo build --release --target-dir "!TARGET_DIR!"
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Echec de la compilation de projet_gaiapi.
    cd /d "%~dp0"
    exit /b %ERRORLEVEL%
)

set "BUILT_DLL=!TARGET_DIR!\release\gaiapi.dll"
if exist "!BUILT_DLL!" (
    echo [OK] gaiapi.dll compilee avec succes.
    copy /y "!BUILT_DLL!" "%~dp0training the model\gaiapi.dll" >nul 2>nul
    if exist "%~dp0gaia_gui_lab" (
        copy /y "!BUILT_DLL!" "%~dp0gaia_gui_lab\gaiapi.dll" >nul 2>nul
    )
    echo [OK] Bibliotheque synchronisee dans 'training the model\' et 'gaia_gui_lab\'.
) else (
    echo [ERREUR] Fichier gaiapi.dll introuvable dans !BUILT_DLL!.
    cd /d "%~dp0"
    exit /b 1
)

cd /d "%~dp0"
echo ============================================================
echo   ✅ Compilation terminee avec succes !
echo ============================================================
