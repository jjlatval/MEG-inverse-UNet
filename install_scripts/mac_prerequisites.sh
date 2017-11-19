#!/usr/bin/env bash

# Run this script to install necessary dependencies for MacOS environment.
# Usage: bash mac_prerequisites.sh --gpu 1

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

echo "Updating Homebrew..."
brew upgrade
brew install python --framework
brew install coreutils
brew install swig

# Install Cask
brew tap caskroom/cask
brew tap caskroom/drivers
brew cask install java
brew install bazel
brew install cmake

# Install VTK
echo "Installing VTK with all requirements..."
brew install python --framework
brew install cartr/qt4/pyqt
brew install homebrew/science/hdf5
brew tap homebrew/science
brew install vtk --python --qt --pyqt

# Use the default brew package location
echo "Linking Homebrew VTK with virtualenv Python..."
VTK_SITE_PACKAGE_DIR=/usr/local/lib/python2.7/site-packages/
rm ${VENV_SITE_PACKAGE_DIR}/homebrew.pth
echo 'import site; site.addsitedir("'${VTK_SITE_PACKAGE_DIR}'")' >> ${VENV_SITE_PACKAGE_DIR}/homebrew.pth

# GPU & CUDA
if [ "$GPU" = 1 ];
    then echo "Installing with GPU enabled...";
    if [ -z ${DYLD_LIBRARY_PATH} ]; then echo "CUDA has been already successfully installed!";
    else
        echo "Installing CUDA..."
        brew cask install cuda

        # Test which CUDA version
        cd /Developer/NVIDIA/
        CUDA_VERSIONS=`find . -maxdepth 1 -regex '.*CUDA-[0-9].[0-9]'`
        ORIG_VERSION=0.0

        for version in ${CUDA_VERSIONS}
            do
            VERSION=$(echo ${version}| cut -d'-' -f2)

                if (( $(echo "$VERSION > $ORIG_VERSION" |bc -l) )); then
                    ORIG_VERSION=${VERSION};
                fi
            done

        CUDA_PATH=/Developer/NVIDIA/CUDA-${ORIG_VERSION}

        echo "export PATH=${CUDA_PATH}"/'bin${PATH:+:${PATH}}' >> ./bash_profile
        echo "export DYLD_LIBRARY_PATH='/usr/local/cuda/lib':"'$DYLD_LIBRARY_PATH' >> ./bash_profile
        echo "CUDA successfully installed. Add environment variables to your bash profile. Reboot recommended.";
    fi
else echo "Not installing GPU dependencies."
fi
