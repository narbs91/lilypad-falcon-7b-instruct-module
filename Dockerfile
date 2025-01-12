FROM --platform=linux/amd64 python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Create outputs directory
RUN mkdir -p /outputs
RUN chmod 777 /outputs

# Copy the inference script
COPY run_inference.py .

# Copy the download script
COPY download_module.py .

RUN python3 download_module.py

# Set outputs directory as a volume
VOLUME /app/outputs

# Run the inference script
CMD ["python", "run_inference.py"]
