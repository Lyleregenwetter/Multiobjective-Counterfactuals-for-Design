FROM python:3.9-slim-buster
WORKDIR /app
RUN apt-get update -y && apt-get upgrade -y && apt-get install libgomp1 -y
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY src/ src/
COPY tests/ tests/
COPY run-tests.sh run-tests.sh
COPY pyproject.toml pyproject.toml
RUN ./run-tests.sh