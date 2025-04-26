FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create uploads directory if it doesn't exist
RUN mkdir -p mobile_price_prediction/data/uploads

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]