FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    MEMORIA_DATA_DIR=/var/lib/memoria

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 memoria \
    && mkdir --parents /var/lib/memoria \
    && chown --recursive memoria:memoria /app /var/lib/memoria

USER memoria

EXPOSE 8080

CMD ["uvicorn", "memoria.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
