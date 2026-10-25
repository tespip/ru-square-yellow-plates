FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src ./src
COPY weights ./weights

ENV ALPR_INPUT_DIR=/data/images ALPR_OUTPUT_CSV=/data/out/result.csv
ENTRYPOINT ["python", "-m", "src.run"]
