#!/bin/bash

# Initialize workspace structure
mkdir -p /workspace/slegospace/{dataspace,recordspace,functionspace,knowledgespace,ontologyspace}

# Start Jupyter Lab
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='easy-token' \
    --NotebookApp.notebook_dir=/workspace \
    --NotebookApp.default_url="/lab/tree/SLEGO.ipynb"