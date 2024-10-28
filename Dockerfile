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

# Create startup script
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=easy-token

EXPOSE 8888 5006

CMD ["/workspace/start.sh"]
