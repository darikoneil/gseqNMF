# flake8: noqa
import os
import subprocess
import sys
from pathlib import Path

import pytest
import toml


"""
A super disgusting way to test the packages install on WINDOWS ONLY. 
I wrote this when I first learned python, but it works so I kepe it around for the lols.
"""
def retrieve_details(path) -> tuple[str, str, list[str]]:
    """
    Get the package details

    :param path: The path to the package's pyproject.toml
    :returns: The name, version, and dependencies of the package
    """
    
    details = toml.load(path).get("project")
    name = details.get("name")
    version = details.get("version")
    dependencies = details.get("dependencies")
    return name, version, dependencies


def retrieve_project_directory(path) -> Path:
    """
    Retrieves the project's directory

    :param path: Child path to project directory
    :returns: The path to the project's directory
    """
    return Path(path).parent


def retrieve_project_file() -> os.PathLike:
    """
    Retrieves the project file (pyproject.toml)

    :returns: The project file
    :raises FileNotFoundError: if not found
    """
    
    project_file = os.path.join(os.getcwd(), "pyproject.toml")
    if not os.path.exists(project_file):
        project_file = os.path.join(Path(os.getcwd()).parent, "pyproject.toml")
    if not os.path.exists(project_file):
        raise FileNotFoundError("Can't find project file")
    return project_file


def collect_project() -> tuple[Path, os.PathLike, str, str, list[str]]:
    """
    Collects the project

    :returns: The project details (directory, file, name, version, dependencies)
    """
    project_file = retrieve_project_file()
    project_directory = retrieve_project_directory(project_file)
    package_name, package_version, package_dependencies = retrieve_details(project_file)
    return project_directory, project_file, package_name, package_version, package_dependencies


# get project information and work from correct directory
proj_dir, proj_file, pkg_name, pkg_version, pkg_dependencies = collect_project()
os.chdir(proj_dir)

print(f"\nInstalling: {pkg_name}=={pkg_version}, from {proj_file} in {proj_dir}")
print(f"Package dependencies: {pkg_dependencies}")


@pytest.mark.parametrize("path", [proj_dir])
def test_install(path):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e ."])
    except subprocess.CalledProcessError as e:
        print(f"{e.output}")
        # This doesn't work on non-windows systems
