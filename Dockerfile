FROM tensorflow/tensorflow:latest
LABEL maintainer "pangjisu <jisu4160@gmail.com>"
WORKDIR /root
RUN apt-get update
RUN apt-get install -y python3-pip
COPY requirments.txt requirments.txt
RUN pip3 install -r requirments.txt
ENTRYPOINT ["python3"]
CMD ["recognize.py"]
