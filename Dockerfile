FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Helps avoid some tokenizers parallel warnings
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_NO_TF=1

WORKDIR /app

# System deps (needed by some python libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
