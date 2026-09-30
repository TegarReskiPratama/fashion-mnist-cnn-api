FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY artifacts/model.keras ./artifacts/model.keras
EXPOSE 8000
CMD ["uvicorn", "fashion_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
