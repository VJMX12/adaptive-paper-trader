FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# data dir holds sqlite db + learner state — mount a volume at /app/data
# (Railway: `railway volume add --mount-path /app/data`; plain Docker:
#  `-v ./data:/app/data`. The VOLUME directive is rejected by Railway.)
# Run as a non-root user; give it /app/data so it can write the DB/state.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser
EXPOSE 8787
CMD ["python", "main.py"]
