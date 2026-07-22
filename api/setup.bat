@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "API_DIR=%~dp0"
for %%I in ("%API_DIR%.") do set "API_ROOT=%%~fI"
for %%I in ("%API_DIR%..") do set "ROOT_DIR=%%~fI"

call :main
set "SETUP_STATUS=%ERRORLEVEL%"
call :cleanup
exit /b %SETUP_STATUS%

:main
set "VERSION_FILE=%API_DIR%versions.env"
if not exist "%VERSION_FILE%" (
	echo Dependency lock is missing: %VERSION_FILE%
	exit /b 1
)
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%VERSION_FILE%") do set "%%A=%%B"

if not exist "%API_DIR%downloads" mkdir "%API_DIR%downloads"

where curl.exe >nul 2>nul
if errorlevel 1 (
	echo curl.exe is required to download dependencies.
	exit /b 1
)

set "TORCH_VARIANT=cpu"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits >nul 2>nul
if not errorlevel 1 set "TORCH_VARIANT=%TORCH_GPU_VARIANT%"
if not "%GADIDAE_TORCH_VARIANT%"=="" set "TORCH_VARIANT=%GADIDAE_TORCH_VARIANT%"
set "TORCH_VARIANT_ALLOWED="
for %%V in (%TORCH_VARIANTS:,= %) do if /i "%%V"=="%TORCH_VARIANT%" set "TORCH_VARIANT_ALLOWED=1"
if not defined TORCH_VARIANT_ALLOWED (
	echo Unsupported LibTorch variant %TORCH_VARIANT%; allowed: %TORCH_VARIANTS%
	exit /b 1
)

if "%GADIDAE_SKIP_TORCH%"=="1" goto torch_ready
if not exist "%API_DIR%libtorch\share\cmake\Torch" (
  set "TORCH_ZIP=%API_DIR%downloads\libtorch-%TORCH_VERSION%-%TORCH_VARIANT%-win.zip"
  set "TORCH_URL=https://download.pytorch.org/libtorch/%TORCH_VARIANT%/libtorch-win-shared-with-deps-%TORCH_VERSION%%%2B%TORCH_VARIANT%.zip"
  echo Downloading LibTorch %TORCH_VERSION% %TORCH_VARIANT%...
  curl.exe --fail --location --retry 3 --output "!TORCH_ZIP!" "!TORCH_URL!" || exit /b 1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!TORCH_ZIP!' '%API_DIR%'" || exit /b 1
) else (
  echo LibTorch already installed.
)
:torch_ready

if not exist "%API_DIR%ninja\ninja.exe" (
	set "NINJA_ZIP=%API_DIR%downloads\ninja-%NINJA_VERSION%-win.zip"
	echo Downloading Ninja %NINJA_VERSION%...
	curl.exe --fail --location --retry 3 --output "!NINJA_ZIP!" "https://github.com/ninja-build/ninja/releases/download/v%NINJA_VERSION%/ninja-win.zip" || exit /b 1
	if not exist "%API_DIR%ninja" mkdir "%API_DIR%ninja"
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!NINJA_ZIP!' '%API_DIR%ninja'" || exit /b 1
) else (
	echo Ninja already installed.
)

if not exist "%API_DIR%nlohmann\include\nlohmann\json.hpp" (
  echo Downloading nlohmann-json %JSON_VERSION%...
  if not exist "%API_DIR%nlohmann\include\nlohmann" mkdir "%API_DIR%nlohmann\include\nlohmann"
  curl.exe --fail --location --retry 3 --output "%API_DIR%nlohmann\include\nlohmann\json.hpp" "https://github.com/nlohmann/json/releases/download/v%JSON_VERSION%/json.hpp" || exit /b 1
)

if not exist "%API_DIR%chess\chess.hpp" (
  echo Downloading chess-library 0.9.4...
  if not exist "%API_DIR%chess" mkdir "%API_DIR%chess"
  curl.exe --fail --location --retry 3 --output "%API_DIR%chess\chess.hpp" "https://raw.githubusercontent.com/Disservin/chess-library/%CHESS_REF%/include/chess.hpp" || exit /b 1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "if ((Get-FileHash '%API_DIR%chess\chess.hpp' -Algorithm SHA256).Hash -ne '%CHESS_SHA256%') { throw 'chess.hpp checksum mismatch' }" || exit /b 1
)

if not exist "%API_DIR%zlib\lib" (
  set "ZLIB_ZIP=%API_DIR%downloads\zlib-%ZLIB_VERSION%.zip"
  echo Downloading and building zlib %ZLIB_VERSION%...
  if not exist "!ZLIB_ZIP!" curl.exe --fail --location --retry 3 --output "!ZLIB_ZIP!" "https://github.com/madler/zlib/archive/refs/tags/v%ZLIB_VERSION%.zip" || exit /b 1
  if exist "%API_DIR%zlib-src" rmdir /s /q "%API_DIR%zlib-src"
  if exist "%API_DIR%zlib-build" rmdir /s /q "%API_DIR%zlib-build"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!ZLIB_ZIP!' '%API_DIR%zlib-unpack'; Move-Item '%API_DIR%zlib-unpack\zlib-%ZLIB_VERSION%' '%API_DIR%zlib-src'; Remove-Item -Recurse -Force '%API_DIR%zlib-unpack'" || exit /b 1
  cmake -S "%API_DIR%zlib-src" -B "%API_DIR%zlib-build" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DZLIB_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="%API_DIR%zlib" || exit /b 1
  cmake --build "%API_DIR%zlib-build" --config Release --target install || exit /b 1
) else (
  echo zlib already installed.
)

if not exist "%API_DIR%hdf5\lib" (
  set "HDF5_ZIP=%API_DIR%downloads\hdf5-%HDF5_VERSION%.zip"
  echo Downloading and building HDF5 %HDF5_VERSION%...
  if not exist "!HDF5_ZIP!" curl.exe --fail --location --retry 3 --output "!HDF5_ZIP!" "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_%HDF5_VERSION%.zip" || exit /b 1
  if exist "%API_DIR%hdf5-src" rmdir /s /q "%API_DIR%hdf5-src"
  if exist "%API_DIR%hdf5-build" rmdir /s /q "%API_DIR%hdf5-build"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!HDF5_ZIP!' '%API_DIR%hdf5-unpack'; Move-Item '%API_DIR%hdf5-unpack\hdf5-hdf5_%HDF5_VERSION%' '%API_DIR%hdf5-src'; Remove-Item -Recurse -Force '%API_DIR%hdf5-unpack'" || exit /b 1
  cmake -S "%API_DIR%hdf5-src" -B "%API_DIR%hdf5-build" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DZLIB_ROOT="%API_DIR%zlib" -DHDF5_USE_ZLIB_STATIC=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DCMAKE_INSTALL_PREFIX="%API_DIR%hdf5" || exit /b 1
  cmake --build "%API_DIR%hdf5-build" --config Release --target install || exit /b 1
) else (
  echo HDF5 already installed.
)

if not exist "%API_DIR%hdf5\lib" exit /b 1

cmake "-DAPI_DIR=%API_ROOT%" "-DTORCH_DIR=%API_ROOT%\libtorch" "-DEXPECTED_TORCH_VARIANT=%TORCH_VARIANT%" -P "%API_ROOT%\verify.cmake" || exit /b 1

echo.
echo Gadus dependencies ready.
echo LibTorch variant: %TORCH_VARIANT%
echo Build: call "%ROOT_DIR%\scripts\build.bat"
exit /b 0

:cleanup
for %%D in (zlib-src zlib-build zlib-unpack hdf5-src hdf5-build hdf5-unpack downloads) do (
	if exist "%API_DIR%%%D" rmdir /s /q "%API_DIR%%%D"
)
exit /b 0
