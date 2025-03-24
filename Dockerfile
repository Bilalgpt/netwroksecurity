FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && pip install -r requirements.txt

# Document that the container uses port 8005
EXPOSE 8005

CMD ["python3", "app.py"]