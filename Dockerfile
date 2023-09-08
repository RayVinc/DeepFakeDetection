FROM python:3.10-slim
#3.8.12 if it doesn't work

WORKDIR /prod
#COPY . .
#everything from here to prod
#then simply exclude what you don't want in the .dockerignore
COPY requirements.txt requirements.txt
COPY DeepFakeDetection DeepFakeDetection
COPY setup.py setup.py
COPY Makefile Makefile

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \
    libglib2.0-0 -y
# To solve: ImportError: libGL.so.1: cannot open shared object file:
# No such file or directory. See: https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo

RUN pip install --upgrade pip
#make sure it's always updated!
RUN pip install -r requirements.txt
#install package dependencies
RUN pip install .
# install everything coming from setup.py

CMD make reset_local_files
#we need that to create the instance for google
CMD uvicorn DeepFakeDetection.api.fast:app --host 0.0.0.0 --port $PORT