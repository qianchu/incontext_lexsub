#run inside the docker nvidia-docker run  --name context2vec-gpu -it -p 8888:8888 -v /home/ql261/simp2trad/:/home/simp2trad/ chainer/chainer:v4.0.0-python2-lqc /bin/bash 

#install optional
apt-get update
apt-get upgrade
apt-get install git
git config --global user.email "hey_flora@126.com"
git config --global user.name "qianchu"

# set up bash shell
# pack python project

if [ ! -d '/home/ql261/.jupyter/' ]; then
    sudo mkdir /home/ql261/.jupyter/
fi
sudo cp /home/context_specialized/jupyter_notebook_config.py /home/ql261/.jupyter/

for pid in $(ps -def | grep jupyter | awk '{print $2}'); do sudo kill -9 $pid; done

export SHELL=/bin/bash
jupyter notebook --ip '*'  --port=8886 --allow-root &
