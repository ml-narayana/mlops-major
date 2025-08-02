FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

RUN python src/train.py && python src/quantize.py

CMD ["python", "src/predict.py"]
