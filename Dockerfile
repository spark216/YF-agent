FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai
ARG INCLUDE_MODELS=false
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt
RUN if [ "$INCLUDE_MODELS" = "true" ]; then pip install --no-cache-dir -r requirements.txt && playwright install --with-deps chromium; fi
COPY *.py ./
COPY utils/ utils/
COPY models/*.py models/
COPY prompts/ prompts/
COPY web/ web/
COPY data/hotel_catalog.xlsx data/hotel_catalog.xlsx
COPY .env.example .env.example
EXPOSE 8765
CMD ["python", "demo_web.py", "--host", "0.0.0.0"]
