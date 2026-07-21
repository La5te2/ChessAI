@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0.") do set "ROOT_DIR=%%~fI"
set "NINJA_DIR=%ROOT_DIR%\api\ninja"
set "NINJA=%NINJA_DIR%\ninja.exe"
set "BUILD_DIR=%ROOT_DIR%\build"

if not "%GADUS_BUILD_DIR%"=="" set "BUILD_DIR=%GADUS_BUILD_DIR%"
if not exist "%NINJA%" (
	echo Ninja is missing. Run api\setup.bat first.
	exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
	set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
	if not exist "!VSWHERE!" (
		echo MSVC Build Tools are missing.
		exit /b 1
	)
	for /f "usebackq tokens=*" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_ROOT=%%I"
	if not defined VS_ROOT (
		echo MSVC x64 Build Tools are missing.
		exit /b 1
	)
	call "!VS_ROOT!\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul || exit /b 1
)

set "PATH=%NINJA_DIR%;%PATH%"
set "VSLANG=1033"
set "TORCH_ARG="
if not "%GADUS_TORCH_DIR%"=="" set "TORCH_ARG=-DGADIDAE_TORCH_DIR=%GADUS_TORCH_DIR%"

cmake -S "%ROOT_DIR%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release %TORCH_ARG% || exit /b 1
cmake --build "%BUILD_DIR%" --parallel || exit /b 1
ctest --test-dir "%BUILD_DIR%" --output-on-failure || exit /b 1

echo Gadus build finished: %BUILD_DIR%\gadus
