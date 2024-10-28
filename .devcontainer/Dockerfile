FROM mcr.microsoft.com/devcontainers/python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    graphviz \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Setup environment variables
ENV PYTHONUNBUFFERED=1