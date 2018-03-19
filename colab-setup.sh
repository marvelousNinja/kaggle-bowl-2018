#/bin/bash!
export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true

apt-get install -y python3.6-tk nano htop screen nvidia-cuda-toolkit --no-install-recommends
pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip3 install -r requirements.txt

./make.sh
