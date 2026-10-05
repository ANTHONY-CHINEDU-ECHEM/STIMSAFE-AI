# Stage 1: build the React dashboard.
FROM node:22-slim AS dashboard
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend ./
RUN npm run build

# Stage 2: Python runtime serving the API and the built dashboard.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY configs ./configs
COPY artifacts ./artifacts
COPY --from=dashboard /frontend/dist ./frontend/dist
RUN useradd --create-home stimsafe && chown -R stimsafe /app
USER stimsafe
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "stimsafe.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
