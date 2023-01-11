FROM python:3.9

ENV PYTHONPATH "${PYTHONPATH}:/code/app"


RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get upgrade -y && apt-get install -y libaio1 unzip ffmpeg libsm6 libxext6 

RUN mkdir -p /opt/oracle && mkdir -p /code

COPY . /code/app

WORKDIR /code/app 


RUN pip install --no-cache-dir --upgrade -r requirements.txt


CMD ["python", "detect.py"]


