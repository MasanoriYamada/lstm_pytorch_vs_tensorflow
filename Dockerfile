FROM ubuntu:16.04

RUN apt-get update && apt-get install -y git && apt-get install -y python3 && apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install torch==1.1.0 tensorflow==1.14.0
