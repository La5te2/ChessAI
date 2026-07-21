@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "API_DIR=%~dp0"
set "ROOT_DIR=%API_DIR%.."
set "TORCH_VERSION=2.13.0"
set "HDF5_VERSION=1.14.6"
set "ZLIB_VERSION=1.3.1"
set "JSON_VERSION=3.12.0"
set "NINJA_VERSION=1.12.1"
set "CHESS_REF=master"

if not "%GADUS_TORCH_VERSION%"=="" set "TORCH_VERSION=%GADUS_TORCH_VERSION%"
if not exist "%API_DIR%downloads" mkdir "%API_DIR%downloads"

set "TORCH_VARIANT=cpu"
where nvidia-smi >nul 2>nul
if %ERRORLEVEL%==0 set "TORCH_VARIANT=cu126"
if not "%GADUS_TORCH_VARIANT%"=="" set "TORCH_VARIANT=%GADUS_TORCH_VARIANT%"

if "%GADUS_SKIP_TORCH%"=="1" goto torch_ready
if not exist "%API_DIR%libtorch\share\cmake\Torch" (
  set "TORCH_ZIP=%API_DIR%downloads\libtorch-%TORCH_VERSION%-%TORCH_VARIANT%-win.zip"
  set "TORCH_URL=https://download.pytorch.org/libtorch/%TORCH_VARIANT%/libtorch-win-shared-with-deps-%TORCH_VERSION%%%2B%TORCH_VARIANT%.zip"
  echo Downloading LibTorch %TORCH_VERSION% %TORCH_VARIANT%...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing '!TORCH_URL!' -OutFile '!TORCH_ZIP!'" || exit /b 1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!TORCH_ZIP!' '%API_DIR%'" || exit /b 1
) else (
  echo LibTorch already installed.
)
:torch_ready

if not exist "%API_DIR%ninja\ninja.exe" (
	set "NINJA_ZIP=%API_DIR%downloads\ninja-%NINJA_VERSION%-win.zip"
	echo Downloading Ninja %NINJA_VERSION%...
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing 'https://github.com/ninja-build/ninja/releases/download/v%NINJA_VERSION%/ninja-win.zip' -OutFile '!NINJA_ZIP!'" || exit /b 1
	if not exist "%API_DIR%ninja" mkdir "%API_DIR%ninja"
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!NINJA_ZIP!' '%API_DIR%ninja'" || exit /b 1
) else (
	echo Ninja already installed.
)

if not exist "%API_DIR%nlohmann\include\nlohmann\json.hpp" (
  echo Downloading nlohmann-json %JSON_VERSION%...
  if not exist "%API_DIR%nlohmann\include\nlohmann" mkdir "%API_DIR%nlohmann\include\nlohmann"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing 'https://github.com/nlohmann/json/releases/download/v%JSON_VERSION%/json.hpp' -OutFile '%API_DIR%nlohmann\include\nlohmann\json.hpp'"
  if errorlevel 1 exit /b 1
)

if not exist "%API_DIR%chess\chess.hpp" (
  echo Downloading chess-library 0.9.4...
  if not exist "%API_DIR%chess" mkdir "%API_DIR%chess"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/Disservin/chess-library/%CHESS_REF%/include/chess.hpp' -OutFile '%API_DIR%chess\chess.hpp'; if ((Get-FileHash '%API_DIR%chess\chess.hpp' -Algorithm SHA256).Hash -ne 'F2C8E2E929641E2C71CBE9D8ABD718CF3CAC46C2A34531215EBD733905E98D7F') { throw 'chess.hpp checksum mismatch' }"
  if errorlevel 1 exit /b 1
)

if not exist "%API_DIR%zlib\lib" (
  set "ZLIB_ZIP=%API_DIR%downloads\zlib-%ZLIB_VERSION%.zip"
  echo Downloading and building zlib %ZLIB_VERSION%...
  if not exist "!ZLIB_ZIP!" powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing 'https://github.com/madler/zlib/archive/refs/tags/v%ZLIB_VERSION%.zip' -OutFile '!ZLIB_ZIP!'" || exit /b 1
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
  if not exist "!HDF5_ZIP!" powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing 'https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_%HDF5_VERSION%.zip' -OutFile '!HDF5_ZIP!'" || exit /b 1
  if exist "%API_DIR%hdf5-src" rmdir /s /q "%API_DIR%hdf5-src"
  if exist "%API_DIR%hdf5-build" rmdir /s /q "%API_DIR%hdf5-build"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!HDF5_ZIP!' '%API_DIR%hdf5-unpack'; Move-Item '%API_DIR%hdf5-unpack\hdf5-hdf5_%HDF5_VERSION%' '%API_DIR%hdf5-src'; Remove-Item -Recurse -Force '%API_DIR%hdf5-unpack'" || exit /b 1
  cmake -S "%API_DIR%hdf5-src" -B "%API_DIR%hdf5-build" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DZLIB_ROOT="%API_DIR%zlib" -DHDF5_USE_ZLIB_STATIC=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DCMAKE_INSTALL_PREFIX="%API_DIR%hdf5" || exit /b 1
  cmake --build "%API_DIR%hdf5-build" --config Release --target install || exit /b 1
) else (
  echo HDF5 already installed.
)

if not exist "%API_DIR%hdf5\lib" exit /b 1

if exist "%API_DIR%zlib-src" rmdir /s /q "%API_DIR%zlib-src"
if exist "%API_DIR%zlib-build" rmdir /s /q "%API_DIR%zlib-build"
if exist "%API_DIR%hdf5-src" rmdir /s /q "%API_DIR%hdf5-src"
if exist "%API_DIR%hdf5-build" rmdir /s /q "%API_DIR%hdf5-build"
if exist "%API_DIR%downloads" rmdir /s /q "%API_DIR%downloads"

echo.
echo Gadus dependencies ready.
echo LibTorch variant: %TORCH_VARIANT%
echo Build: call "%ROOT_DIR%\build.bat"
