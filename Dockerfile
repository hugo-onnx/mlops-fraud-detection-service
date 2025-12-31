# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY --from=builder /app/.venv /app/.venv

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
