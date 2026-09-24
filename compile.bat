@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Projet Gaia - Compilation du Moteur Natif Rust (Windows)
echo ============================================================

where cargo >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Le compilateur Rust 'cargo' n'a pas ete trouve dans le PATH.
    echo Installez Rust depuis https://rustup.rs/ puis reessayez.
    exit /b 1
)

cd "%~dp0projet_gaiapi"
echo Compilant en mode release...
cargo build --release
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Echec de la compilation de projet_gaiapi.
    cd "%~dp0"
    exit /b %ERRORLEVEL%
)

if exist "%~dp0projet_gaiapi\target\release\gaiapi.dll" (
    echo [OK] gaiapi.dll compilee avec succes dans target\release\
    copy /y "%~dp0projet_gaiapi\target\release\gaiapi.dll" "%~dp0training the model\gaiapi.dll" >nul 2>nul
)

cd "%~dp0"
echo ============================================================
echo   Compilation terminee avec succes !
echo ============================================================
