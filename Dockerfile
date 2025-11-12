FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OS libraries Chromium needs (matches your Playwright warning)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl dumb-init \
    libglib2.0-0 libgobject-2.0-0 libnspr4 libnss3 libnssutil3 libsmime3 \
    libgio-2.0-0 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libexpat1 \
    libxcb1 libxkbcommon0 libatspi2.0-0 libx11-6 libxcomposite1 libxdamage1 \
    libxext6 libxfixes3 libxrandr2 libgbm1 libcairo2 libpango-1.0-0 libasound2 \
    fonts-liberation \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m playwright install chromium

COPY . .

EXPOSE 8000
ENV PORT=8000

# Uvicorn binds to ${PORT} if Railway sets it; defaults to 8000
CMD ["dumb-init","bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
