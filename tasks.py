import os
import sys

from invoke import task


@task
def test(c, coverage=False):
    """
    Run tests using pytest.

    Args:
        c: Context object.
        coverage (bool): Whether to run tests with coverage.
    """
    cmd = "uv run pytest -v --tb=short"
    if coverage:
        cmd += " --cov=geopdp --cov-report=term-missing --cov-report=xml"
    c.run(cmd, echo=True)


@task
def lint(c, fix=False):
    """
    Run linting and formatting checks (ruff).

    Args:
        c: Context object.
        fix (bool): Whether to automatically fix issues.
    """
    if fix:
        c.run("uv run ruff format .", echo=True)
        c.run("uv run ruff check --fix .", echo=True)
    else:
        c.run("uv run ruff format --check .", echo=True)
        c.run("uv run ruff check .", echo=True)


@task
def format(c):
    """
    Alias for lint --fix.

    Args:
        c: Context object.
    """
    lint(c, fix=True)


@task
def check_notebooks(c):
    """
    Strip output from notebooks and check for uncommitted changes.

    Args:
        c: Context object.
    """
    notebooks = []
    for root, dirs, files in os.walk("."):
        if ".venv" in dirs:
            dirs.remove(".venv")
        if ".git" in dirs:
            dirs.remove(".git")
        if ".ipynb_checkpoints" in dirs:
            dirs.remove(".ipynb_checkpoints")

        for file in files:
            if file.endswith(".ipynb"):
                notebooks.append(os.path.join(root, file))

    if not notebooks:
        return

    for nb in notebooks:
        c.run(f'uv run nbstripout "{nb}"')

    try:
        c.run('git diff --quiet -- "*.ipynb"')
    except Exception:
        print("Error: Notebook outputs detected. Please clean them before pushing.")
        print(
            "Run 'invoke check-notebooks' locally to fix this "
            "(outputs are stripped, so just commit the changes)."
        )
        sys.exit(1)


@task
def check_requirements(c):
    """
    Check if requirements.txt matches pyproject.toml.

    Args:
        c: Context object.
    """
    if not os.path.exists("requirements.txt"):
        return

    c.run("uv pip compile pyproject.toml -o requirements-check.txt")

    try:
        with (
            open("requirements.txt") as f1,
            open("requirements-check.txt") as f2,
        ):
            reqs = f1.read()
            check = f2.read()

        if reqs != check:
            print("requirements.txt is not up to date with pyproject.toml")
            print("Run: uv pip compile pyproject.toml -o requirements.txt")
            sys.exit(1)

    finally:
        if os.path.exists("requirements-check.txt"):
            os.remove("requirements-check.txt")


@task
def build_docs(c):
    """
    Build documentation (placeholder). Will be filled in once I add sphinx.

    Args:
        c: Context object.
    """
    pass
