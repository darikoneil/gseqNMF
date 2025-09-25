@echo off

:: move to project root
cd ..

:: move to docs folder
cd docs

:: clean previously built docs
call make clean

:: return to root after cleaning
cd ..

:: loop through the possible options for building the documentation
:loop
if "%1"=="source" (goto source)
if "%1"=="html" (goto html)
if "%1"=="rtd" (goto readthedocs)
if "%1"=="" (goto end)

:: build source
:source
echo Building source
call sphinx-apidoc -o docs/source {{ project_name }} -f -e
shift
goto loop

:: build html (local use, not sufficient for readthedocs)
:html
cd docs
echo Building html
call make html
cd ..
shift
goto loop

:: build readthedocs
:readthedocs
echo Building readthedocs
cd docs
call make html
pip freeze > rtd_requirements.txt
python -m truncate_requirements.py
cd ..
shift
goto loop

:: End of building
:end
echo Documentation built
