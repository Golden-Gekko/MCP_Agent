FROM python:3.11-slim

WORKDIR /opt/app

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache

ENV PATH="/opt/app/.venv/bin:$PATH"

COPY . .

CMD ["python", "main.py"]