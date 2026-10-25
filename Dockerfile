FROM continuumio/anaconda3:latest

ENV PROJ_LIB=/opt/conda/share/proj/ SRC=/usr/local/src/ClimVarChange

RUN apt-get update && \
    apt-get install -q -y \
    dvipng texlive texlive-fonts-recommended texlive-lang-cyrillic texlive-lang-english texlive-lang-european texlive-latex-extra

RUN mkdir -p $SRC

COPY requirements.txt $SRC

WORKDIR $SRC

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority true && \
    conda install -y --file requirements.txt

ENTRYPOINT ["python"]