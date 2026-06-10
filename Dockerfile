# SatoshiStacker — headless agent image for a cheap CPU VPS (no GPU).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DB_PATH=/data/satoshistacker.db

WORKDIR /app

# non-root runtime user
RUN useradd -m -u 10001 satoshi

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# only the agent package + the backtest-gate marker (needed for the live gate)
COPY agent/ ./agent/
COPY backtest/results/GATE_PASSED ./backtest/results/GATE_PASSED

RUN mkdir -p /data && chown -R satoshi:satoshi /data /app
USER satoshi

# SIGTERM -> loop.run graceful shutdown (does NOT cancel resting bids)
STOPSIGNAL SIGTERM

# state (SQLite + paper book) persists in the mounted /data volume
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "agent.main"]
CMD ["--mode", "testnet"]
