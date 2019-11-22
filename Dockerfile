# FROM tensorflow/tensorflow:2.0.0-py3
FROM tensorflow/tensorflow:2.0.0-gpu-py3
ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . /app
WORKDIR /app

ENTRYPOINT ["python", "experiment.py"]