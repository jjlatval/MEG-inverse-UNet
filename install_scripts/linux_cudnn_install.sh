#!/usr/bin/env bash

# Run this script to install cuDNN in Linux from downloaded tar file
# Usage: bash linux_cudnn_install.sh ~/Downloads/cudnn-8.0-linux-x64-v6.0.tgz

file="$1"

if [ -f ${file} ]; then

    # Test which CUDA version
    cd /usr/local/
    CUDA_VERSIONS=`find . -maxdepth 1 -regex '.*cuda-[0-9].[0-9]'`
    ORIG_VERSION=0.0

    for version in ${CUDA_VERSIONS}
        do
        VERSION=$(echo ${version}| cut -d'-' -f2)

            if (( $(echo "$VERSION > $ORIG_VERSION" |bc -l) )); then
                ORIG_VERSION=${VERSION};
            fi
        done

    CUDA_PATH=/usr/local/cuda-${ORIG_VERSION}

    f_path=$(dirname "${file}")

    tar zxvf ${file} -C /cudnn
    sudo cp -P /cudnn/cuda/include/cudnn.h ${CUDA_PATH}/include
    sudo cp -P cudnn/cuda/lib64/libcudnn* ${CUDA_PATH}/lib64
    sudo chmod a+r ${CUDA_PATH}/lib64/libcudnn*

    sudo apt-get install libcupti-dev

    echo "cuDNN successfully installed.";
else
    echo "File not found";
fi
