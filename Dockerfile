FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (g++ required for annoy package, curl for Node.js)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x for promptfoo
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json and install npm dependencies for promptfoo
COPY package.json .
RUN npm install

# Copy application
COPY app/ ./app/
COPY config/ ./config/

# Copy promptfoo configuration
COPY promptfoo/ ./promptfoo/

# Expose ports
EXPOSE 8000
EXPOSE 15500

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
