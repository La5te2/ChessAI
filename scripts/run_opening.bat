@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
pushd "%ROOT_DIR%" || exit /b 1

if /i "%~1"=="-h" goto help
if /i "%~1"=="--help" goto help

if not defined PGN set "PGN=data\games.pgn"
if not defined MIN_FENS set "MIN_FENS=50000"
if not defined OUTPUT set "OUTPUT=data\openings.gen.bin"
if not defined UCI set "UCI=models\stockfish\stockfish.exe"
if not defined MAX_ABS_CP set "MAX_ABS_CP=80"
if not defined BOOK_PLIES set "BOOK_PLIES=8"
if not defined UCI_DEPTH set "UCI_DEPTH=10"
if not defined UCI_MOVETIME_MS set "UCI_MOVETIME_MS=0"
if not defined UCI_THREADS set "UCI_THREADS=4"
if not defined UCI_HASH_MB set "UCI_HASH_MB=512"
if not defined LOG_EVERY set "LOG_EVERY=1000"

if not "%~1"=="" set "PGN=%~1"
if not "%~2"=="" set "MIN_FENS=%~2"
if not "%~3"=="" set "OUTPUT=%~3"

if not exist "%PGN%" (
	echo Opening PGN is missing: %PGN%
	popd
	exit /b 1
)
if not exist "%UCI%" (
	echo UCI engine is missing: %UCI%
	popd
	exit /b 1
)

set "PYTHON=python"
where python >nul 2>nul
if errorlevel 1 set "PYTHON=py -3"

echo Opening generation start
echo Opening source: pgn=%PGN%
echo Opening output: output=%OUTPUT% min_fens=%MIN_FENS% book_plies=%BOOK_PLIES%
echo Opening engine: uci=%UCI% uci_depth=%UCI_DEPTH% uci_movetime_ms=%UCI_MOVETIME_MS% uci_threads=%UCI_THREADS% uci_hash_mb=%UCI_HASH_MB%
echo Opening filter: max_abs_cp=%MAX_ABS_CP% log_every=%LOG_EVERY%

%PYTHON% scripts\opening_book.py ^
	--pgn "%PGN%" ^
	--uci "%UCI%" ^
	--output "%OUTPUT%" ^
	--max-abs-cp "%MAX_ABS_CP%" ^
	--book-plies "%BOOK_PLIES%" ^
	--min-fens "%MIN_FENS%" ^
	--uci-depth "%UCI_DEPTH%" ^
	--uci-movetime-ms "%UCI_MOVETIME_MS%" ^
	--uci-threads "%UCI_THREADS%" ^
	--uci-hash-mb "%UCI_HASH_MB%" ^
	--log-every "%LOG_EVERY%"
set "STATUS=%ERRORLEVEL%"
popd
exit /b %STATUS%

:help
echo usage: scripts\run_opening.bat [pgn] [min_fens] [output_bin]
echo example: scripts\run_opening.bat data\games.pgn 50000 data\openings.gen.bin
popd
exit /b 0
