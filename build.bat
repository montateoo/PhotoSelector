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
echo.
echo [2/3] Building .exe (this takes a minute)...
pyinstaller ^
    --onefile ^
    --windowed ^
    --icon=icon.ico ^
    --add-data "icon.ico;." ^
    --name=PhotoSelector ^
    --hidden-import=PyQt5.sip ^
    main.py
if errorlevel 1 ( echo ERROR: PyInstaller failed & pause & exit /b 1 )

:: 3. Copy .exe to project root for convenience
echo.
echo [3/3] Copying output...
copy /y dist\PhotoSelector.exe PhotoSelector.exe

echo.
echo Build complete!
echo Output: %cd%\PhotoSelector.exe
echo.
pause
