FROM python:3.11-slim

WORKDIR /app
COPY python_backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY python_backend ./python_backend
COPY data ./data
COPY models ./models

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--chdir", "python_backend", "--bind", "0.0.0.0:8080", "app:app"]
