FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "cleanflow_env.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
