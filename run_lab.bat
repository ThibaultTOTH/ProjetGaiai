@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Projet Gaia - Lancement de Gaia AI Lab & Sandbox (Windows)
echo ============================================================

set SCRIPT_DIR=%~dp0
set DLL_PATH=%SCRIPT_DIR%projet_gaiapi\target\release\gaiapi.dll

if not exist "!DLL_PATH!" (
    echo [!] gaiapi.dll introuvable. Lancement de la compilation...
    call "%SCRIPT_DIR%compile.bat"
    if %ERRORLEVEL% neq 0 (
        echo [ERREUR] Impossible de compiler gaiapi.dll.
        pause
        exit /b 1
    )
)

echo [✓] Moteur Rust pret.
echo [+] Lancement de l'interface graphique et de l'evaluateur neuronal...

set KMP_DUPLICATE_LIB_OK=TRUE
cd "%SCRIPT_DIR%gaia_gui_lab"
python main.py
if %ERRORLEVEL% neq 0 (
    echo [!] Erreur lors de l'execution avec 'python'. Tentative avec 'py'...
    py -3 main.py
)

cd "%SCRIPT_DIR%"
