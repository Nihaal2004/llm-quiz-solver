FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OS libs Chromium needs on Debian (trixie/bookworm compatible)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl dumb-init \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libcairo2 libcups2 \
    libdbus-1-3 libdrm2 libexpat1 libgbm1 libglib2.0-0 libgtk-3-0 \
    libnspr4 libnss3 libpangocairo-1.0-0 libpango-1.0-0 \
    libwayland-client0 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 \
    libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxkbcommon0 \
    libxrandr2 libxrender1 libxshmfence1 fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m playwright install chromium

# app
COPY . .

# Optional: many PaaS run Chrome as root; no-sandbox avoids permission issues
# If your code doesn't already pass it, add this in main.py when launching:
#   browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])

EXPOSE 8000
ENV PORT=8000
CMD ["dumb-init","bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
