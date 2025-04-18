FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Flask application
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]