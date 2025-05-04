FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy everything into container
COPY . . 

# Expose port for MLflow
EXPOSE 5001

# Serve the model
CMD ["mlflow", "models", "serve", "-m", "models:/churn_model/1", "--no-conda", "-h", "0.0.0.0", "-p", "5001"]