FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn pandas numpy

COPY train.py /app/train.py

ENTRYPOINT ["python", "-u", "/app/train.py"]
