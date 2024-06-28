FROM python:3.8-slim-buster
LABEL maintainer="Mohsin Aslam"
LABEL email="m.mohsin.aslam@gmail.com"
LABEL version="1.0"

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

# Install locales package and ensure that the locale is generated.
RUN apt-get update -yqq && \
    apt-get upgrade -yqq && \
    apt-get install -yqq locales && \
    locale-gen en_US.UTF-8

# Set environment variables
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Install basic packages
RUN set -ex \
    && apt-get install -yqq --no-install-recommends \
        curl \
        git \
        locales \
        netcat \
        build-essential \
        libpq-dev \
    # Cleanup unnecessary files
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    # Install Python dependencies
    && pip install -U pip setuptools wheel \
    # Install psycopg2-binary for PostgreSQL support
    && pip install psycopg2-binary

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
