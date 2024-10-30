# Use an official Python 3.10 base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make build-essential

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run your Panel web application
CMD ["python", "-m", "panel", "serve", "app.py"]
