FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# data dir holds sqlite db + learner state; mount it as a volume
VOLUME ["/app/data"]
EXPOSE 8787
CMD ["python", "main.py"]
