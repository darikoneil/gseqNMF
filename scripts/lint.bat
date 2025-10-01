@echo off
setlocal enabledelayedexpansion

:: move to project root
cd ..

echo [0;33m "Formatting (RUFF)..." [0m
ruff format ./gseqNMF ./tests

echo [0;33m "Linting (RUFF)..." [0m
ruff check ./gseqNMF ./tests -o .ruff.json --output-format json --fix --no-cache

:: too opinionated for linting tests
echo [0;33m "Linting (FLAKE8 PLUGINS)..." [0m
flake8 ./gseqNMF

echo [0;33m "Finished Formatting & Linting" [0m
endlocal
