# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Set entrypoint
CMD ["python", "main.py"]
