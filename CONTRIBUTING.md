# Contributing to geopdp

Thank you for your interest in contributing to `geopdp`! We welcome contributions that help make geospatial machine learning interpretation easier and more robust.

Please take a moment to review our design philosophy to ensure your contributions align with the project's goals.

## Design Philosophy

**`geopdp` prioritizes explicitness, transparency, and predictability over "magic" convenience.**

When adding new features or modifying existing ones, adhere to the following principles:

### 1. Explicit is Better Than Implicit
*   **Do not assume column names.** Always require users to specify which columns correspond to latitude, longitude, region, etc.
*   **Do not hardcode paths.** Users must explicitly provide paths to external resources (e.g., GeoJSON files).
*   **Avoid "magic" behavior.** If a function's behavior changes significantly based on the presence of an optional argument, consider splitting it into two functions.

### 2. No Hidden Defaults
*   **Avoid defaults that affect results.** Defaults should only be used for harmless stylistic choices (e.g., plot size) that do not alter the data or the interpretation.

### 3. Fail Fast and Loud
*   **Raise specific exceptions early.** If a required argument is missing or invalid (e.g., `None` where a value is needed), raise a `ValueError` or `KeyError` immediately.
*   **Avoid silent fallbacks.** Do not silently substitute missing values or guess user intent. If the input is ambiguous, ask the user to clarify via arguments.

### 4. Zero Global State
*   Functions should be pure whenever possible. Avoid relying on or modifying global variables or settings.

## Development Workflow

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Run tests**:
    ```bash
    uv run pytest
    ```

3.  **Linting**:
    ```bash
    uv run ruff check .
    ```

## Pull Requests

*   Ensure all new features are covered by tests.
*   Update docstrings and type hints for any changed functions.
*   If your change modifies the API, update `README.md` and `tests/` to reflect the new explicit requirements.
