@echo off
setlocal EnableExtensions

if "%~2"=="" goto :usage

set "ARCH=%~1"
set "MODEL_ARG=%~2"

if /I not "%ARCH%"=="gadus" if /I not "%ARCH%"=="melano" if /I not "%ARCH%"=="eleginus" (
	echo Unsupported architecture: %ARCH%
	exit /b 2
)

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
pushd "%ROOT%" || exit /b 1

for %%I in ("%MODEL_ARG%") do set "MODEL=%%~fI"
if /I "%ARCH%"=="eleginus" goto :eleginus
set "UCI=%ROOT%\build\%ARCH%\uci.exe"
set "OUTPUT=%ROOT%\models\%ARCH%"

if not exist "%MODEL%" (
	echo Model not found: %MODEL%
	popd
	exit /b 3
)
if not exist "%UCI%" (
	echo UCI executable not found: %UCI%
	echo Build first with: scripts\build.bat
	popd
	exit /b 4
)

if not exist "%OUTPUT%" mkdir "%OUTPUT%"
copy /Y "%UCI%" "%OUTPUT%\%ARCH%.exe" >nul || goto :copy_error
if /I not "%MODEL%"=="%OUTPUT%\%ARCH%.pth" (
	copy /Y "%MODEL%" "%OUTPUT%\%ARCH%.pth" >nul || goto :copy_error
)
for %%F in ("%ROOT%\build\%ARCH%\*.dll") do (
	copy /Y "%%~fF" "%OUTPUT%\%%~nxF" >nul || goto :copy_error
)

echo Gadidae UCI engine packaged
echo architecture=%ARCH%
echo executable=%OUTPUT%\%ARCH%.exe
echo checkpoint=%OUTPUT%\%ARCH%.pth
echo UCI command=%OUTPUT%\%ARCH%.exe
popd
exit /b 0

:eleginus
set "TYPE=%~3"
if "%TYPE%"=="" set "TYPE=uci"
if /I not "%TYPE%"=="uci" if /I not "%TYPE%"=="search" (
	echo Eleginus package type must be uci or search: %TYPE%
	popd
	exit /b 2
)
set "EMBED=%ROOT%\build\eleginus\embed.exe"
set "OUTPUT=%ROOT%\models\eleginus"
if /I "%TYPE%"=="uci" (
	set "ENGINE=%OUTPUT%\eleginus.exe"
) else (
	set "ENGINE=%OUTPUT%\eleginus_search.exe"
)
if not exist "%MODEL%" (
	echo Model not found: %MODEL%
	popd
	exit /b 3
)
if not exist "%EMBED%" (
	echo Eleginus embed executable not found: %EMBED%
	echo Build first with: scripts\build.bat
	popd
	exit /b 4
)
if not exist "%OUTPUT%" mkdir "%OUTPUT%"
"%EMBED%" --model "%MODEL%" --type "%TYPE%" --output "%ENGINE%" || goto :copy_error
echo Gadidae Eleginus executable packaged
echo type=%TYPE%
echo executable=%ENGINE%
popd
exit /b 0

:copy_error
echo Failed to write engine package: %OUTPUT%
popd
exit /b 5

:usage
echo Usage: scripts\package_engine.bat ^<gadus^|melano^|eleginus^> ^<model.pth^> [uci^|search]
echo Example: scripts\package_engine.bat gadus models\gadus\candidate3.pth
echo Example: scripts\package_engine.bat eleginus models\eleginus\eleginus.pth uci
exit /b 1
