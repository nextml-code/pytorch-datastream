FROM python:3.8

WORKDIR /usr/src/pytorch-datastream

COPY pytest.ini ./
COPY requirements.txt ./
COPY scripts/ ./scripts/
COPY setup.* ./

ENV TERM vt100
RUN virtualenv venv -p python3.8
RUN echo "source venv/bin/activate" >> ${HOME}/.bashrc

COPY datastream/ ./datastream
COPY .git/ ./.git/
RUN bash -c 'source venv/bin/activate && pip install . && python -c "import datastream"'

ENTRYPOINT [ "bash" ]
