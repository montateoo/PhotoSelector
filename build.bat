@echo off
setlocal

echo.
echo === PhotoSelector build ===
echo.

:: 1. Generate icon
echo [1/3] Generating icon...
python make_icon.py
if errorlevel 1 ( echo ERROR: icon generation failed & pause & exit /b 1 )

:: 2. Build .exe with PyInstaller
::    Video is decoded with OpenCV (bundles its own ffmpeg) — no Qt multimedia
::    or OS codecs required.
::    Clean build/ and dist/ first: PyInstaller's incremental cache can silently
::    skip re-collecting cv2 binaries, producing an exe with no video support.
echo.
echo [2/3] Building .exe (this takes a minute)...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
set "SECRETS="
if exist secrets.toml (set "SECRETS=--add-data secrets.toml;.") else (echo WARNING: secrets.toml not found - crash reports will be disabled)
pyinstaller ^
    --onefile ^
    --windowed ^
    --icon=icon.ico ^
    --add-data "icon.ico;." ^
    %SECRETS% ^
    --name=PhotoSelector ^
    --hidden-import=PyQt5.sip ^
    --collect-all PyQt5 ^
    --hidden-import=cv2 ^
    --collect-all cv2 ^
    main.py
if errorlevel 1 ( echo ERROR: PyInstaller failed & pause & exit /b 1 )

:: 4. Copy .exe to project root for convenience
echo.
echo [3/3] Copying output...
copy /y dist\PhotoSelector.exe PhotoSelector.exe

echo.
echo Build complete!
echo Output: %cd%\PhotoSelector.exe
echo.
pause
