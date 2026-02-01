FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (g++ required for annoy package, libxcb for PDF processing)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxcb1 \
    libxcb-render0 \
    libxcb-render-util0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install Python dependencies
# Use --no-cache-dir and clean up pip cache to save space
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && rm -rf /tmp/* /var/tmp/*

# Copy application
COPY app/ ./app/
COPY config/ ./config/

# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
