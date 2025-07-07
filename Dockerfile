# Use Python slim image for ARM
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train_model.py .
COPY inference.py .

# Train the model during build
RUN python train_model.py

# Set the default command
CMD ["python", "inference.py"]

