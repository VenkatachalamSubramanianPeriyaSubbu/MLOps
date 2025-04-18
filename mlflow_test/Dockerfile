# Start from a basic Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install mlflow
RUN pip install mlflow

# Expose port 5000 so we can access the mlflow UI
EXPOSE 5000

# Command to run the mlflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
