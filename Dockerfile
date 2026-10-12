# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download the uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Never re-resolve dependencies at runtime; copy instead of hardlink across layers
ENV UV_FROZEN=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Set the working directory
WORKDIR /code

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock /code/

# Install dependencies using uv.
#
# Two things are baked into this layer rather than fetched at run time:
#   - en_core_web_lg, the 382 MB spaCy model, declared as a project dependency
#   - torch and torchvision, resolved from the PyTorch CPU index on Linux
#     (see [tool.uv.sources] in pyproject.toml), which avoids pulling in about
#     2.5 GB of CUDA libraries that a CPU container cannot use
#
# The layer is cached as long as pyproject.toml and uv.lock do not change.
RUN uv sync --frozen --no-dev

# Copy the application code, including the trained CNN weights in app/weights.
# Doing this after the dependency layer means code edits do not trigger a
# reinstall of the dependencies.
COPY ./app /code/app

EXPOSE 80

# Command to run the application
CMD ["uv", "run", "--frozen", "fastapi", "run", "app/main.py", "--port", "80"]
