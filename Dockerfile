FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY configs ./configs
COPY src ./src
COPY webapp ./webapp

USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
