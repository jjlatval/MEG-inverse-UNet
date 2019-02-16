## MEG inverse problem U-net estimator

This repository contains a neural network model capable of estimating the MEG inverse problem
with a modified U-net convolutional network (see: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) by using L2 MNE, sLORETA, dSPM and/or raw sensor data as input. This repository is the tool I used when writing my thesis on improving the L2 minimum norm based solutions by "post-processing" them with a neural network. My thesis can be found here: http://urn.fi/URN:NBN:fi:aalto-201712188173

I am aware that there are most likely way more efficient ways of addressing the MEG inverse problem with a neural network. If I had to start again from scratch, I would most likely try to find a regression model that would converge (numerous LSTMs were tested during development). Anyway, have fun with the repository!

#### General Requirements

* Ubuntu or MacOS (64 bit)
    * Tested on Ubuntu 16.04 LTS (x86_64) and MacOS Sierra (x86_64)
    * x86_64 architecture required for GPU acceleration with cuDNN
* Python 2.7+/3.6+ (or Anaconda2/3)
* pip: `https://pip.pypa.io/en/stable/installing/`
* virtualenv: `sudo pip install virtualenv`
* At least 100 GB of free disk space
* Powerful enough PC
    * CUDA Compute 3.0 capable NVidia GPU with the latest drivers if you use GPU for training (highly recommended)
        * cuDNN 5.1 is needed. A newer version does not seem to work with the newest Tensorflow
    * Enough RAM memory: Even the simplest simulations may take ~20 GB of working memory
    * Training times (XX per epoch)
* Tested on the following machine:
    * Ubuntu PC
        * OS: Ubuntu 16.04 LTS (64-bit)
        * CPU: 3.2 Ghz Octa-core Intel Xeon E5 2667V4
        * RAM: 32 GB
        * GPU: NVidia Geforce 1080 Ti
        * Training the U-Net with ico3 subdivision takes 235 epochs and roughly 30 hours with this setup

#### MacOS Prerequisite steps:

1. Install homebrew: `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
2. Install XCode command line tools: `xcode-select --install`
3. Create virtualenv `virtualenv venv`
    * If you use Python 3 you can use `-p python3` option
        * Also use pip3 if you use Python 3
4. Activate virtualenv: `source venv/bin/activate`
5. Install prerequisites:
    1. (OPTION 1) With NVidia CUDA 3.0 capable GPU (only x86_64 architecture required):
        1. Install Xcode by typing `gcc` in terminal
        2. Install MacOS prerequisites with CUDA: `bash mac_prerequisites.sh --gpu 1` *
        *) using --gpu 1 makes the installating script to install CUDA with required environment variables. If you want to do it yourself of if you encounter issues with it, you can try to install without --gpu
        3. Register & download cuDNN 5.1 from (https://developer.nvidia.com/cudnn)
        4. Untar and install downloaded file by with: `bash install_scripts/mac_cudnn_install.sh ~/Downloads/cudnn-8.0-osx-x64-v6.0.tgz`
    2. (OPTION 2) Without GPU support:
        1. Install MacOS prerequisites without CUDA: `bash install_scripts/mac_prerequisites.sh`
6. Define matplotlib backend renderer. Add "backend: TkAgg" to ~/.matplotlib/matplotlibrc: `echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc`
    
###### Note: Tensorflow does not support GPU acceleration on MacOS from version 1.2 onwards
* It seems that training the network works perfectly with using tensorflow-gpu==1.1.0 whereas predicing with the network and running Tensorboard does not work in MacOS with tensorflow-gpu. You can use tensorflow-gpu for training and `pip install tensorflow` when you need to predict with the network or use Tensorboard.
* Another solution is to have two virtualenvironments: one for tensorflow-gpu and one for tensorflow

#### Linux Prerequisite steps:

1. Create virtualenv `virtualenv venv`
2. Activate virtualenv: `source venv/bin/activate`
3. Install prerequisites:
    1. (OPTION 1) With NVidia CUDA 3.0 capable GPU (x86_64 architecture required):
        1. Install Linux prerequisites with CUDA: `bash install_scripts/linux_prerequisites.sh --gpu 1`
            1. If you want to specify a CUDA version, you can use cuda argument: `bash install_scripts/linux_prerequisites.sh --cuda 8.0.61-1 --gpu 1`
        2. Register & download cuDNN 5.1 from (https://developer.nvidia.com/cudnn)
    2. (OPTION 2) Without GPU support:
        1. Install Linux prerequisites without CUDA: `bash install_scripts/linux_prerequisites.sh`
        
#### Getting started

1. Make sure that your virtualenv is still active
2. Install FreeSurfer and setup its environment variables: (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) (needed for visualization)
3. Install requirements in your virtualenv: `pip install -r gpu-requirements.txt` OR `pip install -r requirements.txt` for non-GPU usage
    * For Python 3 `pip3 install -r gpu-requirements.txt` OR `pip3 install -r requirements.txt`
    * NOTE: installing requirements may take a while. Especially when PySide builds Qt in the background.
4. Install Tensorflow Unet: `bash install_tf_unet.sh`
5. Run the network:
    1. Simulate data and train the network yourself
        1. Configure config.py parameters to suit your computing resources and research needs
        2. Create training, validation and testing datasets: `python simulate_data.py`
        3. Train the network: `python train_network.py`

#### Testing and visualizing

Diagnostics folder contains some scripts for testing the generated dataset. These include dataset visualization and calculating class frequencies in order to detect class imbalances. In diagnostics folder you can:

1. Run predictions: `python predict.py`
2. Clean the .csv file in network_tests folder and save the cleaned file as `results_processed.csv`
3. Run basic statisics by running: `r analyze_data.r` (NOTE: install R with some additional libraries: TODO add these dependencies somewhere)
4. Visualize simulated data `python visualize_simulated_data.py` (only for 1 dipole case)


#### Contribution guidelines ###

* Anyone can contribute. Right now help is needed with:
    * Writing tests
    * Code review
* I update the repository when issues emerge. Note that the quality of the newer code is worse due to DL pressures. I am not actively developing this repository further but may do so if the codebase is actually used for something useful. There is some old code as well as TODO markets. If you want, you can improve the codebase. You are free to create pull requests for fixing issues, and I will gladly review them.

#### Any questions? ###

* Mail me at joni.latvala[at]gmail.com

#### Credits

* The codebase is heavily based on MNE-Python, so big thanks to you developers
* The Unet model is based on tf_unet by Joel Akeret (https://media.readthedocs.org/pdf/tf-unet/latest/tf-unet.pdf)
