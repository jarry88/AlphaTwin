FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN pip install jupyterlab

# Create directories
RUN mkdir -p data/raw data/processed src notebooks

CMD ["python"]
