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
        jupyter \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    panel \
    param \
    pyvis \
    rdflib \
    networkx \
    kglab

# Convert SLEGO.ipynb to app.py
RUN jupyter nbconvert --to script SLEGO.ipynb --output app.py

# Expose the port for Panel application
EXPOSE 5006

# Run the app.py with Panel
CMD ["panel", "serve", "/workspace/app.py", "--allow-websocket-origin=*"]
