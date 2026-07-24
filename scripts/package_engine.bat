@echo off
setlocal EnableExtensions

if "%~2"=="" goto :usage

set "ARCH=%~1"
set "MODEL_ARG=%~2"

if /I not "%ARCH%"=="gadus" if /I not "%ARCH%"=="melano" (
	echo Unsupported architecture: %ARCH%
	exit /b 2
)

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
pushd "%ROOT%" || exit /b 1

for %%I in ("%MODEL_ARG%") do set "MODEL=%%~fI"
set "UCI=%ROOT%\build\%ARCH%\uci.exe"
set "OUTPUT=%ROOT%\models\gadidae"

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
copy /Y "%MODEL%" "%OUTPUT%\%ARCH%.pth" >nul || goto :copy_error
for %%F in ("%ROOT%\build\%ARCH%\*.dll") do (
	copy /Y "%%~fF" "%OUTPUT%\%%~nxF" >nul || goto :copy_error
)

echo Gadidae UCI engine packaged
echo architecture=%ARCH%
echo executable=%OUTPUT%\%ARCH%.exe
echo checkpoint=%OUTPUT%\%ARCH%.pth
echo Cute Chess command=%OUTPUT%\%ARCH%.exe
popd
exit /b 0

:copy_error
echo Failed to write engine package: %OUTPUT%
popd
exit /b 5

:usage
echo Usage: scripts\package_engine.bat ^<gadus^|melano^> ^<model.pth^>
echo Example: scripts\package_engine.bat gadus models\gadus\candidate3.pth
exit /b 1
