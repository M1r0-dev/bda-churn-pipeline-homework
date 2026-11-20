FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt
COPY --chmod=644 pipeline.py config.json ./
COPY data/ ./data/
RUN mkdir -p /app/artifacts
CMD ["python", "pipeline.py", "--config", "config.json"]
