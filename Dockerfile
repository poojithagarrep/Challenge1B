# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.10-slim

# Install system dependencies required by pdfplumber (poppler-utils)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy all files to /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt


# Default command to run the system
CMD ["python", "main_runner.py"]
