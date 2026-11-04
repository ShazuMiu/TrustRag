@echo off
REM Offline end-to-end check on Windows: ingest -> tests -> benchmark -> report
cd /d "%~dp0\.."
set CFG=%1
if "%CFG%"=="" set CFG=config/lite.yaml
python -m trustrag.cli --config %CFG% ingest --input data/demo/corpus || exit /b 1
python -m unittest discover -s tests || exit /b 1
python -m trustrag.cli --config %CFG% eval --dataset demo --out results/demo_run || exit /b 1
echo Open results\demo_run\report.html
