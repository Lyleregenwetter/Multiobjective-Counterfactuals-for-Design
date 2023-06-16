FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update -y && apt-get upgrade -y && apt-get install libgomp1 -y
COPY src/ src/
COPY tests/ tests/
RUN ./run-tests.sh