# syntax=docker/dockerfile:1

# This image serves the Streamlit dashboard only. It reads pre-computed
# parquet files from data/ — it never recomputes anything and never
# imports settings.py, so it never needs ALPHA_VANTAGE_API_KEY or any
# other secret. The daily pipeline that DOES need that key runs in
# GitHub Actions (.github/workflows/pipeline.yml), not in this container.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry \
    && poetry install --only main \
    && rm -rf "$(poetry config cache-dir)"

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
