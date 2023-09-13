FROM tensorflow/tensorflow:2.10.0
#python:3.10-slim
#3.8.12 if it doesn't work

WORKDIR /prod

#COPY . .
# copy everything from here to prod then simply exclude what you don't want in the .dockerignore
COPY requirements.txt requirements.txt
COPY DeepFakeDetection DeepFakeDetection
COPY setup.py setup.py
COPY Makefile Makefile

RUN pip install --upgrade pip
#make sure it's always updated!
RUN pip install -r requirements.txt
#install package dependencies
RUN pip install .
# install everything coming from setup.py

CMD make reset_local_files
#we need that to create the instance for google
CMD uvicorn DeepFakeDetection.api.fast:app --host 0.0.0.0 --port $PORT
