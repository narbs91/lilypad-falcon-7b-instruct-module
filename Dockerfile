FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create outputs directory
RUN mkdir -p outputs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script
COPY run_inference.py .

# Set outputs directory as a volume
VOLUME /app/outputs

# Run the inference script
CMD ["python", "run_inference.py"]
