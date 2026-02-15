# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/
COPY dags/ /app/dags/ 
# Copying dags as well just in case, though usually airflow runs separately.
# For the inference API, we strictly need src and models.

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.apps.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
