@echo off
setlocal

set "ROOT=%~dp0\..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "WORK=%ROOT%\build\.pyinstaller-gui"

cd /d "%ROOT%"

python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
	echo PyInstaller is required.
	echo Install it with: python -m pip install pyinstaller
	exit /b 1
)

if exist "%WORK%" rmdir /s /q "%WORK%"
mkdir "%WORK%"

echo Building api\gui\Gadidae.exe...
python -m PyInstaller ^
	--noconfirm ^
	--clean ^
	--onefile ^
	--windowed ^
	--name Gadidae ^
	--exclude-module pygame ^
	--exclude-module numpy ^
	--exclude-module torch ^
	--paths "%ROOT%\api\gui" ^
	--distpath "%ROOT%\api\gui" ^
	--workpath "%WORK%\work" ^
	--specpath "%WORK%\spec" ^
	"%ROOT%\api\gui\gadidae.py"

if errorlevel 1 (
	echo GUI build failed. Diagnostics retained in: %WORK%
	exit /b 1
)

rmdir /s /q "%WORK%"
echo Gadidae GUI finished: %ROOT%\api\gui\Gadidae.exe
endlocal
