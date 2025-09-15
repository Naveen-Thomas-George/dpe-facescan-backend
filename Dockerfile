# 1. Start from an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy the requirements file and install dependencies with a longer timeout
COPY requirements.txt .
# --- THIS IS THE UPDATED LINE ---
# We've added --default-timeout=300 to give pip up to 5 minutes for each download
RUN pip --default-timeout=300 install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code into the container
COPY . .

# 7. Tell Docker that the container will listen on port 8000
EXPOSE 8000

# 8. Define the command to run when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

