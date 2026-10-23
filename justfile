# uv provisions the pinned interpreter and locked environment automatically.
check:
    uv run --locked --group examples ruff check .
    uv run --locked --group examples ruff format --check .
    uv run --locked --group examples pyrefly check
    just test

test *args:
    uv run --locked --group examples pytest -n auto -q {{args}}

test-algorithms *args:
    uv run --locked pytest --confcutdir=tests/algorithms tests/algorithms -q {{args}}

# Build the package wheel under dist/.
build:
    uv build --wheel --out-dir dist

alias wheel := build
