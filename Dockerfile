FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04
WORKDIR /app
RUN apt update && apt install -fy python3 python3-pip && apt clean
COPY requirements.txt /app
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt
COPY . /app
ENTRYPOINT ["python3","/app/bot.py"]
