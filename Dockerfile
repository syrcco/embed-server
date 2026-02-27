FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch first (separate layer for caching — largest dependency)
RUN pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir zentorch==5.1.0 sentence-transformers fastapi uvicorn

# Copy application code
COPY server.py /app/

# OMP_NUM_THREADS=4 matches RS1000's 4 vCPU allocation
ENV OMP_NUM_THREADS=4

EXPOSE 8765

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]
