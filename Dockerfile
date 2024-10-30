# Start from a base Python image
FROM mcr.microsoft.com/devcontainers/python:3.10

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        curl \
        graphviz \
        build-essential \
        nodejs \
        npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    jupyter \
    jupyterlab \
    panel \
    param \
    pyvis \
    rdflib \
    networkx \
    kglab

# Expose the port used by Panel
EXPOSE 5006

# Run the app
CMD ["panel", "serve", "/workspace/app.py", "--allow-websocket-origin=*", "--port", "5006"]
