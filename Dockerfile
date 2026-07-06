FROM python:3.12-slim
WORKDIR /app
# gosu lets the entrypoint start as root (to fix volume ownership) then drop
# to a non-root user before running the app.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# data dir holds sqlite db + learner state — mount a volume at /app/data
# (Railway: `railway volume add --mount-path /app/data`; plain Docker:
#  `-v ./data:/app/data`. The VOLUME directive is rejected by Railway.)
# A mounted volume overrides build-time ownership, so the entrypoint fixes it
# at runtime and then drops privileges — the app never runs as root.
RUN useradd --create-home --uid 10001 appuser
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
EXPOSE 8787
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "main.py"]
