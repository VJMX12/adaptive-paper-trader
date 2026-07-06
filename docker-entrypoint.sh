#!/bin/sh
# Start as root, make the (possibly volume-mounted) data dir writable by the
# non-root user, then drop privileges and exec the app. The app process itself
# never runs as root.
set -e
mkdir -p /app/data
chown -R appuser:appuser /app/data 2>/dev/null || true
exec gosu appuser "$@"
