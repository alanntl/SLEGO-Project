#!/bin/bash

# Start Jupyter Lab without token and auto-open SLEGO.ipynb
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.notebook_dir=/workspace \
    --NotebookApp.default_url="/lab/tree/SLEGO.ipynb" &

# Wait for Jupyter to start
sleep 5

# Auto-execute the notebook to start the web app
jupyter nbconvert --to notebook --execute SLEGO.ipynb --inplace

# Keep container running
tail -f /dev/null