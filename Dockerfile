# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set the working directory
WORKDIR /app

# 3. INSTALL SYSTEM DEPENDENCIES (The Fix)
# We need to update apt and install the missing GL libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your specific folders
COPY pipeline/ ./pipeline/
COPY weights/ ./weights/
COPY environment/ ./environment/
COPY model_card/ ./model_card/
COPY input_data/ ./input_data/
COPY output_data/ ./output_data/
COPY logs/ ./logs/

# 7. Run the app
CMD ["python", "pipeline/main.py"]