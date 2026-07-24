@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "NINJA_DIR=%ROOT_DIR%\api\ninja"
set "NINJA=%NINJA_DIR%\ninja.exe"
set "TORCH_DIR=%ROOT_DIR%\api\libtorch"
set "PUBLISH_DIR=%ROOT_DIR%\build"
set "WORK_DIR=%PUBLISH_DIR%\.build-work"

if not "%GADIDAE_TORCH_DIR%"=="" set "TORCH_DIR=%GADIDAE_TORCH_DIR%"
if not exist "%NINJA%" (
	echo Ninja is missing. Run api\setup.bat first.
	exit /b 1
)
if not exist "%TORCH_DIR%\share\cmake\Torch\TorchConfig.cmake" (
	echo LibTorch is missing or GADIDAE_TORCH_DIR is invalid.
	exit /b 1
)
cmake -DAPI_DIR="%ROOT_DIR%\api" -DTORCH_DIR="%TORCH_DIR%" -P "%ROOT_DIR%\api\verify.cmake" || exit /b 1

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

for %%I in ("%WORK_DIR%") do set "RESOLVED_WORK=%%~fI"
for %%I in ("%PUBLISH_DIR%") do set "RESOLVED_PUBLISH=%%~fI"
if /i not "%RESOLVED_WORK%"=="%ROOT_DIR%\build\.build-work" exit /b 1
if /i not "%RESOLVED_PUBLISH%"=="%ROOT_DIR%\build" exit /b 1

if not exist "%PUBLISH_DIR%" mkdir "%PUBLISH_DIR%" || exit /b 1
set "PATH=%NINJA_DIR%;%PATH%"
set "VSLANG=1033"

cmake -S "%ROOT_DIR%" -B "%WORK_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGADIDAE_TORCH_DIR="%TORCH_DIR%" || goto :failed
cmake --build "%WORK_DIR%" --parallel || goto :failed
ctest --test-dir "%WORK_DIR%" --output-on-failure || goto :failed

if exist "%PUBLISH_DIR%\gadus" rmdir /s /q "%PUBLISH_DIR%\gadus"
if exist "%PUBLISH_DIR%\melano" rmdir /s /q "%PUBLISH_DIR%\melano"
mkdir "%PUBLISH_DIR%\gadus" || goto :failed
mkdir "%PUBLISH_DIR%\melano" || goto :failed

for %%A in (gadus melano) do (
	for %%F in (preprocess train search arena fcpi uci) do (
		if not exist "%WORK_DIR%\%%A\%%F.exe" goto :failed
		copy /y "%WORK_DIR%\%%A\%%F.exe" "%PUBLISH_DIR%\%%A\%%F.exe" >nul || goto :failed
	)
	for %%F in ("%WORK_DIR%\%%A\*.dll") do (
		copy /y "%%~fF" "%PUBLISH_DIR%\%%A\%%~nxF" >nul || goto :failed
	)
)

echo Gadus build finished: %PUBLISH_DIR%\gadus
echo Melano build finished: %PUBLISH_DIR%\melano
echo Incremental build cache: %WORK_DIR%
exit /b 0

:failed
set "ERROR_CODE=%ERRORLEVEL%"
if "%ERROR_CODE%"=="0" set "ERROR_CODE=1"
if exist "%WORK_DIR%" (
	echo Build failed. Diagnostic files retained in: %WORK_DIR%
	if exist "%WORK_DIR%\Testing\Temporary\LastTest.log" (
		echo CTest log: %WORK_DIR%\Testing\Temporary\LastTest.log
	)
)
exit /b %ERROR_CODE%
