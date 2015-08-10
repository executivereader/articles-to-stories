#!/usr/bin/env bash
# the contents of this file should be put in the user data field
sudo apt-get update
sudo apt-get install -y awscli git python-dev python-pip libopenblas-dev
sudo mkfs -t ext4 /dev/xvdf
sudo mkdir /data
sudo mount /dev/xvdf /data
cd /home/ubuntu/
sudo git clone https://github.com/executivereader/mongo-startup.git
sudo git clone https://github.com/executivereader/articles-to-stories.git
sudo cp /home/ubuntu/mongo-startup/connection_string.txt /home/ubuntu/articles-to-stories/connection_string.txt
sudo cp /home/ubuntu/mongo-startup/update_replica_set.py /home/ubuntu/articles-to-stories/update_replica_set.py
cd /home/ubuntu/articles-to-stories
sudo wget https://s3.amazonaws.com/word2vec-googlenews/GoogleNews-vectors-negative300.bin.gz
sudo pip install numpy scipy
sudo pip install gensim pymongo
sudo screen -dm bash -c "sudo python articles-to-stories.py"
