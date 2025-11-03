# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY --from=builder /root/.cache/uv /root/.cache/uv
COPY . .
EXPOSE 8080
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]