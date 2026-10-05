    
## Installation
All operations should be done on machines with GPUs.

```
git clone https://github.com/JinlongW128/CoDS.git
cd CoDS
```
### 1. Install conda
Please refer to https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

### 2. Create conda environment and set up the base dependencies
```
conda create --name CoDS python=3.8
conda activate CoDS
conda install cudnn -c conda-forge
conda install boost

# install pytorch
pip install torch==1.12.0+cu116 \
    torchvision==0.13.0+cu116 \
    torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

##### *(Option) If there is error or speed issues in install cudatoolkit
```
# could instead specify the PATH, CUDA_HOME, and LD_LIBRARY_PATH, using current cuda write it to ~/.bashrc, for example use Vim
vim ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda/bin:$CUDA_HOME
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# add head file search directories 
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/Anaconda3/envs/CoDS/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/Anaconda3/envs/CoDS/include
# add shared library searching directories
export LIBRARY_PATH=$LIBRARY_PATH:/Anaconda3/envs/CoDS/lib
# add runtime library searching directories
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Anaconda3/envs/CoDS/lib

# go out of Vim and activate it in current shell
source ~/.bashrc

conda activate CoDS
```

### 3. Install spconv (Support both 1.2.1 and 2.x)

##### *(Notice): Make sure *libboost-all-dev* is installed in your linux system before installing *spconv*. If not:
```
sudo apt-get install libboost-all-dev
```

##### Install v1.2.1
```
# clone spconv:
git clone https://github.com/traveller59/spconv.git 
cd spconv
git checkout v1.2.1
git submodule update --init --recursive

# compile
python setup.py bdist_wheel

# install
cd ./dist
pip install spconv-1.2.1-cp37-cp37m-linux_x86_64.whl

# check if is successfully installed
python 
import spconv
```


##### Install 2.x
```
pip install spconv-cu116==2.3.6
```

### 4. Install pypcd
```
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
```

### 5. Install CoDS
```
# install requirements
pip install -r requirements.txt
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace

# FPVRCNN's iou_loss dependency (optional)
python opencood/pcdet_utils/setup.py build_ext --inplace
```
##### *(Option) If there is cuda version issue; ssh db92 -p 58122 and customize the cuda home
```
CUDA_HOME=/usr/local/cuda-11.1/ python opencood/pcdet_utils/setup.py build_ext --inplace
```

### 6. *(Option) for training and testing SCOPE&How2comm
```
# install basic library of deformable attention
git clone https://github.com/TuSimple/centerformer.git
cd centerformer

# install requirements
pip install -r requirements.txt
sh setup.sh
```

##### if there is a problem about cv2:
```
# module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
pip install opencv-python install "opencv-python-headless<4.3"
```
