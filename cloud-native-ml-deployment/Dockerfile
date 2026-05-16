FROM python:3.12-slim

WORKDIR /app

COPY mnist_model.keras /app/
COPY app.py /app/

RUN pip install --no-cache-dir \
tensorflow \
flask \
numpy

EXPOSE 5000

CMD ["python", "app.py"]
