FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --frozen

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

EXPOSE 8080

CMD ["uv", "run", "python", "scripts/run_api.py"]
