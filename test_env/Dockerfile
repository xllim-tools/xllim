FROM jupyter/scipy-notebook 
RUN apk update && apk add gcc python3-dev
MAINTAINER Sami Djouadi

ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /app
WORKDIR /app
COPY ./test_env /app

RUN adduser -D user
USER user

CMD /launch.sh
