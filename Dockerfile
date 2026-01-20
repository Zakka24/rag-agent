FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    tesseract-ocr \
    tesseract-ocr-ita \
    poppler-utils \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --no-cache-dir --index-url https://download.pytorch.org/whl/cu130

RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/deepdoctection/detectron2.git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]