#!/usr/bin/env bash

# Run this script to install necessary dependencies for Linux environment.
# Usage: bash linux_prerequisites.sh --cuda 8.0 --gpu 1

if [ -z $VIRTUAL_ENV ]; then
    echo "Virtualenv not active, please active virtualenv."
    exit 1;
fi

PYTHON_VER=`python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";`
VENV_SITE_PACKAGE_DIR=${VIRTUAL_ENV}/lib/python${PYTHON_VER}/site-packages

# Get options
while [[ $# -gt 1 ]]
    do
    key="$1"
        case $key in
            -c|--cuda)
            CUDA_VER="$2"
            shift
            ;;
            -g|--gpu)
            GPU="$2"
            shift # past argument
            ;;
            *)
            echo "Unknown option used."
            exit 1
            ;;
        esac
        shift # past argument or value
    done

# Update apt-get and install requirements
echo "Updating apg-get & installing requirements..."
sudo apt-get update
sudo apt-get upgrade
sudo apt-get -f install
sudo apt-get install build-essential cmake gcc git unzip pkg-config
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install linux-image-generic linux-image-extra-virtual
sudo apt-get install linux-source linux-headers-generic
sudo apt-get install python-wxgtk3.0 python-qt4 python-pyqt5 python-matplotlib python-setuptools python-tables mayavi2

# Install VTK
sudo apt-get install vtk5.10 libvtk5-dev python-vtk

cp -r /usr/lib/python"${PYTHON_VER}"/dist-packages/vtk ${VENV_SITE_PACKAGE_DIR}

# GPU & CUDA

if [ "$GPU" = 1 ];
    then echo "Installing with GPU enabled...";
    if [ -z ${CUDA_HOME} ]; then echo "CUDA has been already successfully installed!";
    else
        OS=`echo "$(lsb_release -si)" | tr '[:upper:]' '[:lower:]'`
        ARCH=$(uname -m)

        if [ "$ARCH" == "x86_64" ]; then ARCH="amd_64";
        else
            echo "Architecture ${ARCH} not supported."
            exit 1;
        fi

        VER_BASE=$(lsb_release -sr)
        OS_VER=echo ${VER_BASE//.}
        CUDA_VER=${CUDA_VER:=8.0.61-1}

        echo "Installing CUDA ${CUDA_VER}..."

        sudo apt-get install linux-headers-$(uname -r)

        CUDA_INSTALL="cuda-repo-${OS}_${OS_VER}_${CUDA_VER}_${ARCH}.deb"
        sudo dpkg -i ${CUDA_INSTALL}

        sudo apt-get update
        sudo apt-get install cuda

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

        Add environment variables into ~/.bashrc
        echo 'Setting environment variables to ~/.bashrc ...'
        echo '# CUDA Toolkit' >> ~/.bashrc
        echo "export CUDA_HOME=${CUDA_PATH}" >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH' >> ~./bashrc
        echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~./bashrc

        source ~/.bashrc
        sudo apt install nvidia-cuda-toolkit
        sudo apt-get install libcupti-dev

        echo "CUDA successfully installed. Reboot recommended.";
    fi
    else echo "Not installing GPU dependencies."
    GPU=0;
fi

sudo ldconfig
