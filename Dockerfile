# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code and artifacts
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.apps.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
