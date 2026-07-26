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
set "BUILD_GRAPHICS=ON"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits >nul 2>nul
if not errorlevel 1 set "TORCH_VARIANT=%TORCH_GPU_VARIANT%"
if not "%GADIDAE_TORCH_VARIANT%"=="" set "TORCH_VARIANT=%GADIDAE_TORCH_VARIANT%"
if /I "%GADIDAE_BUILD_GRAPHICS%"=="0" set "BUILD_GRAPHICS=OFF"
if /I "%GADIDAE_BUILD_GRAPHICS%"=="OFF" set "BUILD_GRAPHICS=OFF"
if /I "%GADIDAE_BUILD_GRAPHICS%"=="FALSE" set "BUILD_GRAPHICS=OFF"
if /I "%GADIDAE_BUILD_GRAPHICS%"=="1" set "BUILD_GRAPHICS=ON"
if /I "%GADIDAE_BUILD_GRAPHICS%"=="ON" set "BUILD_GRAPHICS=ON"
if /I "%GADIDAE_BUILD_GRAPHICS%"=="TRUE" set "BUILD_GRAPHICS=ON"
if not "%GADIDAE_BUILD_GRAPHICS%"=="" if /I not "%GADIDAE_BUILD_GRAPHICS%"=="AUTO" if /I not "%GADIDAE_BUILD_GRAPHICS%"=="0" if /I not "%GADIDAE_BUILD_GRAPHICS%"=="OFF" if /I not "%GADIDAE_BUILD_GRAPHICS%"=="FALSE" if /I not "%GADIDAE_BUILD_GRAPHICS%"=="1" if /I not "%GADIDAE_BUILD_GRAPHICS%"=="ON" if /I not "%GADIDAE_BUILD_GRAPHICS%"=="TRUE" (
	echo GADIDAE_BUILD_GRAPHICS must be auto, 0, or 1.
	exit /b 1
)
set "TORCH_VARIANT_ALLOWED="
for %%V in (%TORCH_VARIANTS:,= %) do if /i "%%V"=="%TORCH_VARIANT%" set "TORCH_VARIANT_ALLOWED=1"
if not defined TORCH_VARIANT_ALLOWED (
	echo Unsupported LibTorch variant %TORCH_VARIANT%; allowed: %TORCH_VARIANTS%
	exit /b 1
)

if "%GADIDAE_SKIP_TORCH%"=="1" goto torch_ready
if not exist "%API_DIR%libtorch\share\cmake\Torch" (
  set "TORCH_ZIP=%API_DIR%downloads\libtorch-%TORCH_VERSION%-%TORCH_VARIANT%-win.zip"
  set "TORCH_URL=https://download.pytorch.org/libtorch/%TORCH_VARIANT%/%TORCH_WINDOWS_ARTIFACT%-%TORCH_VERSION%%%2B%TORCH_VARIANT%.zip"
  echo Downloading LibTorch %TORCH_VERSION% %TORCH_VARIANT%...
  curl.exe --fail --location --retry 3 --output "!TORCH_ZIP!" "!TORCH_URL!" || exit /b 1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!TORCH_ZIP!' '%API_DIR%'" || exit /b 1
) else (
  echo LibTorch already installed.
)
:torch_ready
cmake -DAPI_DIR="%API_ROOT%" -DTORCH_DIR="%API_ROOT%\libtorch" -P "%API_ROOT%\patch.cmake" || exit /b 1

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

if "%BUILD_GRAPHICS%"=="ON" (
	call :download_source glfw "%GLFW_VERSION%" "https://github.com/glfw/glfw/archive/refs/tags/%GLFW_VERSION%.zip" "glfw-%GLFW_VERSION%" || exit /b 1
	call :download_source glm "%GLM_VERSION%" "https://github.com/g-truc/glm/archive/refs/tags/%GLM_VERSION%.zip" "glm-%GLM_VERSION%" || exit /b 1
	call :download_source imgui "%IMGUI_VERSION%" "https://github.com/ocornut/imgui/archive/refs/tags/v%IMGUI_VERSION%.zip" "imgui-%IMGUI_VERSION%" || exit /b 1
	call :download_source freetype "%FREETYPE_VERSION%" "https://github.com/freetype/freetype/archive/refs/tags/VER-%FREETYPE_VERSION%.zip" "freetype-VER-%FREETYPE_VERSION%" || exit /b 1
	powershell -NoProfile -ExecutionPolicy Bypass -Command "$path='%API_DIR%freetype\CMakeLists.txt'; $text=[IO.File]::ReadAllText($path); $text=$text.Replace('cmake_minimum_required(VERSION 3.0...3.5)','cmake_minimum_required(VERSION 3.10...3.31)'); [IO.File]::WriteAllText($path,$text,[Text.UTF8Encoding]::new($false))" || exit /b 1

	if not exist "%API_DIR%glad\include\glad\gl.h" (
		echo Generating GLAD %GLAD_VERSION% for OpenGL 3.3 Core...
		call :download_source glad-generator "%GLAD_VERSION%" "https://github.com/Dav1dde/glad/archive/refs/tags/v%GLAD_VERSION%.zip" "glad-%GLAD_VERSION%" || exit /b 1
		if not exist "%API_DIR%glad-python\jinja2" (
			python -m pip install --disable-pip-version-check --no-cache-dir --target "%API_DIR%glad-python" "Jinja2==%JINJA2_VERSION%" "MarkupSafe==%MARKUPSAFE_VERSION%" || exit /b 1
		)
		set "PYTHONPATH=%API_DIR%glad-python;%API_DIR%glad-generator"
		python -m glad --out-path "%API_DIR%glad" --api gl:core=3.3 --extensions "" --reproducible c --loader || exit /b 1
	)
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

cmake "-DAPI_DIR=%API_ROOT%" "-DTORCH_DIR=%API_ROOT%\libtorch" "-DEXPECTED_TORCH_VARIANT=%TORCH_VARIANT%" "-DVERIFY_GUI=%BUILD_GRAPHICS%" -P "%API_ROOT%\verify.cmake" || exit /b 1

echo.
echo Gadus dependencies ready.
echo LibTorch variant: %TORCH_VARIANT%
echo Graphics: %BUILD_GRAPHICS%
echo Build: call "%ROOT_DIR%\scripts\build.bat"
exit /b 0

:download_source
set "DEP_NAME=%~1"
set "DEP_VERSION=%~2"
set "DEP_URL=%~3"
set "DEP_FOLDER=%~4"
if exist "%API_DIR%!DEP_NAME!\CMakeLists.txt" exit /b 0
set "DEP_ZIP=%API_DIR%downloads\!DEP_NAME!-!DEP_VERSION!.zip"
echo Downloading !DEP_NAME! !DEP_VERSION!...
if not exist "!DEP_ZIP!" curl.exe --fail --location --retry 3 --output "!DEP_ZIP!" "!DEP_URL!" || exit /b 1
if exist "%API_DIR%!DEP_NAME!-unpack" rmdir /s /q "%API_DIR%!DEP_NAME!-unpack"
if exist "%API_DIR%!DEP_NAME!" rmdir /s /q "%API_DIR%!DEP_NAME!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '!DEP_ZIP!' '%API_DIR%!DEP_NAME!-unpack'; Move-Item -LiteralPath '%API_DIR%!DEP_NAME!-unpack\!DEP_FOLDER!' -Destination '%API_DIR%!DEP_NAME!'; Remove-Item -LiteralPath '%API_DIR%!DEP_NAME!-unpack' -Recurse -Force" || exit /b 1
exit /b 0

:cleanup
for %%D in (zlib-src zlib-build zlib-unpack hdf5-src hdf5-build hdf5-unpack glfw-unpack glm-unpack imgui-unpack freetype-unpack glad-generator-unpack glad-generator glad-python downloads) do (
	if exist "%API_DIR%%%D" rmdir /s /q "%API_DIR%%%D"
)
exit /b 0
