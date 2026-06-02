FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uvx /usr/local/bin/uvx

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache

ENV PATH="/opt/app/.venv/bin:$PATH"

COPY . .

CMD ["python", "main.py"]