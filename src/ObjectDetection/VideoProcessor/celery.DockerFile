FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENV CELERY_BROKER_URL=redis://redis:6380/0
ENV CELERY_RESULT_BACKEND=redis://redis:6380/0

CMD ["celery", "-A", "video_processor", "worker", "--loglevel=info"]