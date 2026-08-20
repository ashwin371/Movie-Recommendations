FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the SQLite database at image build time so startup is fast.
RUN python -m src.data_loader || true

EXPOSE 5000

# Serve with gunicorn (production WSGI server).
CMD ["gunicorn", "--chdir", "app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
