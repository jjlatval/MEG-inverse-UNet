#!/usr/bin/env bash

# Run this script to install cuDNN in Mac OS environment from downloaded tar file
# Usage: bash mac_cudnn_install.sh ~/Downloads/cudnn-8.0-osx-x64-v6.0.tgz

file="$1"

if [ -f ${file} ]; then

    f_path=$(dirname "${file}")

    tar zxvf ${file}
    sudo mv -v ${f_path}/cuda/lib/libcudnn* /usr/local/cuda/lib
    sudo mv -v ${f_path}/cuda/include/cudnn.h /usr/local/cuda/include
    sudo rm -rf ${f_path}/cuda
    echo "cuDNN successfully installed.";
else
    echo "File not found";
fi
