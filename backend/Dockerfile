FROM python:3.12-slim

WORKDIR /app

# Create unprivileged user
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

# Create data directory with proper ownership
RUN mkdir -p /app/data && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
