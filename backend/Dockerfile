FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "9", "--timeout", "120", "--bind", "0.0.0.0:5000", "backend:app"]