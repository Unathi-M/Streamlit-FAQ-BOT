# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential

# Copy requirements
COPY requirements_rag.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements_rag.txt

# Copy the project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "rag/app_conv.py", "--server.port=8501", "--server.address=0.0.0.0"]
