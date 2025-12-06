# Pre-commit Hooks Setup

This project uses [pre-commit](https://pre-commit.com/) to automatically run checks before each commit and push. This is to ensure that the code is clean and follows the project's style guide, in addition to the github workflow checks.

## What the hooks do

### On Commit (pre-commit stage)
The following hooks run automatically before each commit:
- **Clear notebook outputs** (`nbstripout`) - Strips outputs from Jupyter notebooks to keep git history clean
- **Lint code** (`ruff`) - Checks for code quality issues and auto-fixes them with `--fix`
- **Format code** (`ruff-format`) - Applies consistent code formatting. Ruff is a faster alternative to black. For a more detailed comparison, see [this](https://github.com/astral-sh/ruff#comparison-with-black).
- **Trailing whitespace** - Removes trailing whitespace from files
- **End-of-file fixer** - Ensures files end with a newline
- **Check YAML** - Validates YAML file syntax
- **Check large files** - Prevent adding files larger than 1MB. This is to prevent large files from being added to the repository. Exceptions are allowed for large files that are not expected to change often, such as large datasets or images.
- **Check merge conflicts** - Detect unresolved merge conflict markers

### On Push (pre-push stage)
The following hooks run automatically before each push:
- **Run tests** (`pytest`) - Execute pytest to ensure all tests pass. This is to catch any potential issues before they are merged.

## Installation

### First time setup

1. Install the development dependencies (includes pre-commit):
   ```bash
   uv sync
   ```

2. Install the git hooks (for both commit and push stages):
   ```bash
   uv run pre-commit install
   uv run pre-commit install --hook-type pre-push
   ```

That's it! The hooks will now run automatically on every commit and push.

## Usage

### Automatic (recommended)
The hooks run automatically when you commit and push:
```bash
git add .
git commit -m "Your commit message"
# Pre-commit hooks run automatically here

git push
# Pre-push hooks (tests) run automatically here
```

If any hook fails, the commit or push will be aborted. Fix the issues and try again.

### Manual run
To run all pre-commit hooks on all files manually:
```bash
uv run pre-commit run --all-files
```

To run a specific hook:
```bash
uv run pre-commit run nbstripout --all-files   # Clear notebook outputs
uv run pre-commit run ruff --all-files         # Lint code
uv run pre-commit run ruff-format --all-files  # Format code
uv run pre-commit run trailing-whitespace --all-files  # Fix trailing whitespace
```

To run the pre-push tests manually:
```bash
uv run pytest
```

### Skip hooks
If you need to skip the hooks for a specific commit:
```bash
git commit --no-verify -m "Your message"
```

To skip pre-push hooks:
```bash
git push --no-verify
```

**Warning**: Only skip hooks when absolutely necessary (e.g., work-in-progress commits). If you are wondering if your commit qualifies, it probably doesn't.

## Updating hooks

To update the pre-commit hooks to the latest versions:
```bash
uv run pre-commit autoupdate
```

## Troubleshooting

### Hooks not running
Make sure you've installed them for both stages:
```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### Tests failing
Run tests manually to see the full output:
```bash
uv run pytest -v
```

### Notebook outputs keep appearing
The `nbstripout` hook should clear them automatically. If not, you can run it manually:
```bash
uv run pre-commit run nbstripout --all-files
```

### Ruff formatting or linting issues
Run ruff manually with verbose output:
```bash
uv run ruff check . --fix
uv run ruff format .
```
