@echo off

if "%1"=="nobuild" (goto publish)

:: build
call build.bat

:: upload to pypi
:publish
cd ..
python -m twine upload --repository pypi dist/* -u __token__ -p %SECRET_TOKEN%
