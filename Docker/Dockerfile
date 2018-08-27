
FROM allennlp/allennlp:v0.5.1
#RUN groupadd -g 999 ql261 && \
    #useradd -r -u 999 -g ql261 ql261

#USER ql261
RUN apt-get update
#RUN apt-get upgrade -y
RUN apt-get install -y \
    sudo  \
    git  \
    vim
RUN useradd -m ql261 && echo "ql261:ql261" | chpasswd && adduser ql261 sudo

USER ql261
