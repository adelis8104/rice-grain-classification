# Use the official Python 3.9 slim image as a base
FROM python:3.9-slim

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libopencv-dev \
      && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt /app/requirements.txt
COPY src/ /app/src
# Data directories will be mounted at runtime, not baked into the image,

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Entrypoint to main script
ENTRYPOINT ["python", "-m", "src.project"]
