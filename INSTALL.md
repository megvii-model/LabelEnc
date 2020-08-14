### Requirements:
- PyTorch >= 1.0. Installation instructions can be found in https://pytorch.org/get-started/locally/.
- torchvision
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9,< 6.0
- (optional) OpenCV for the webcam demo

### Installation:

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name LabelEnc
conda activate LabelEnc

# this installs the right pip and dependencies for the fresh python
conda install ipython

# FCOS and coco api dependencies
pip install ninja yacs cython matplotlib tqdm

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

export INSTALL_DIR=$PWD

# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install LabelEnc
cd $INSTALL_DIR
git clone https://github.com/megvii-model/LabelEnc.git
cd LabelEnc

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop --no-deps


unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```
