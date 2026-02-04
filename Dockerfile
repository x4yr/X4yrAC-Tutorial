FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируй всё из model/ (включая pytorch_model.bin после обучения)
COPY . .

EXPOSE 7860

ENV PORT=7860
CMD ["python", "inference.py"]
