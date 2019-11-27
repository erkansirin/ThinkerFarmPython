# ThinkerFarmPython V1.0   

## Introduction   
This repository contains codes that i used in my previous projects. I gather them together as re-usable python modules in case i needed in the future.

## Hardware
This project developed on Intel Neural Stick 2 (Myriad™ X) and Raspberry Pi 3 B+ Raspbian OS is 32bit, Nvidia Jetson Nano, Intel Up developer kit and for Laptop and Desktop computers if your CPU have AVX instructions (Advanced Vector Extensions) you will get pretty much decent performance in every module. Most of modules are supports GPU by default.  

## Features  

[x] - Face detection with SSD used "res10_300x300_ssd_iter_140000.caffemodel" detection model. *(CPU, VPU, GPU)*  
[x] - Face detection with Haar Cascade *(CPU)*  
[x] - Face recognition with 128D vector  *(CPU, VPU, GPU)*  
[x] - Face recognition with Dlib  *(CPU, GPU)*  
[x] - SSD Object detection with Yolo *(CPU, GPU)*  
[x] - SSD Object detection with MobileNet *(CPU, VPU, GPU)*  
[x] - SSD Object detection with MobileNet + SSD Face detection + CNN Pickle Face recognition + IpCam *(CPU, VPU, GPU)*  
[x] - SSD Object detection with Faster RCNN Tensorflow *(CPU, GPU)*  
[x] - Pose Estimation with OpenPose COCO  *(CPU, VPU, GPU)*  
[x] - Pose Estimation with OpenPose MPI (lite version) *(CPU, VPU, GPU)*  
[] - Pose Estimation with OpenPose with SSD for multiple people *(CPU, VPU, GPU)* (on going)  
[x] - Pose Estimation with OpenPose MPI IpCam (lite version) *(CPU, VPU, GPU)*  
[x] - Pose Estimation with OpenPose Body25 *(CPU, VPU, GPU)* (on going)  
[x] - Pose Estimation with OpenPose HAND *(CPU, VPU, GPU)* (on going)  
[x] - Edge detection *(CPU, VPU, GPU)* (on going)  
[x] - Image classifier module builded on top of OpenCV DNN module and it support all model formats currently DeepScale SqueezeNet implemented *(CPU, GPU)*  
[] - GANs *(CPU, GPU)* (on going)  
[x] - Training : Face recognition model training with Pickle  *(CPU, VPU, GPU)*  
[x] - Training : Face recognition model training Dlib  *(CPU, VPU, GPU)*  
[] - Training : Face recognition model training SSD  *(CPU, VPU, GPU)* (on going)  
[] - Training : SSD object detection model training with Caffe and Tensorflow *(CPU, VPU, GPU)* (on going)  
[] - Training : SSD object detection model training with Caffe and Tensorflow *(CPU, VPU, GPU)* (on going)  

*Note : Some features are not VPU compatible because development environment in Temp-V1.0 is 32bit Raspbian OS OpenVino environment does not support 32bit OS. Also some model files need to be compiled for Intel Neural Stick 2*  

# Installation   

# Auto-install with installation scripts  
## Installing on Ubuntu 18 Jetson Nano              
### On Jetson Nano CUDA packages are comes with default Linux Ubuntu 18 OS images so you don't have to install CUDA just following script it will install all dependencies and download / compile and prepare required frameworks.  

#### Please note that complete setup may take 2-4 hours depending on your internet connection speed


# Manual Installation (Tested on Ubuntu 18)

## Update packages

```
$ sudo apt-get update
``` 

### Installing BLAS and LAPACK packages  

```
$ sudo apt-get update  
$ sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev
```

### Install Dependencies

```
$ sudo apt-get install build-essential cmake unzip pkg-config  
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev  
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev  
$ sudo apt-get install libxvidcore-dev libx264-dev  
$ sudo apt-get install libgtk-3-dev  
$ sudo apt-get install libatlas-base-dev gfortran  
```

### Install and setup Pyhton and virtual environment (optional but recommended)

```
$ sudo apt-get install python-dev python-pip python3-dev python3-pip  
$ sudo apt-get install python3-dev python3-pip
```  

### Project Dependencies installation  

```
$ sudo -H pip3 install pillow  
$ sudo apt-get install python3-tk  
$ sudo pip3 install imutils  
$ sudo pip3 install sklearn  
```


#### imutils compile from source

```
$ git clone https://github.com/jrosebr1/imutils
$ cd imutil
$ sudo python3 setup.py install
```

## Install Dlib

### Clone Dlib  

```
git clone https://github.com/davisking/dlib  
```

### Install for GPU

```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build . --config Release
$ sudo make install
$ sudo ldconfig
$ cd ..
$ sudo python3.6 setup.py install
```

### install for CPU

```
$ mkdir build  
$ cd build  
$ sudo cmake .. -DUSE_AVX_INSTRUCTIONS=ON  
$ sudo cmake --build . --config Release  
$ sudo make install  
$ sudo ldconfig  
$ cd ..
$ sudo python3.6 setup.py install
```

## Install OpenCV Ubuntu 18
```
$ sudo pip3 install virtualenv virtualenvwrapper  
$ sudo rm -rf ~/get-pip.py ~/.cache/pip  
```
To finish the install we need to update our  ~/.bashrc  file.  

Using a terminal text editor such as vi / vim  or nano , add the following lines to your ~/.bashrc :

### virtualenv and virtualenvwrapper
```
$ export WORKON_HOME=$HOME/.virtualenvs  
$ export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3  
$ source /usr/local/bin/virtualenvwrapper.sh  
```
then run command below  

```
$ source ~/.bashrc  
```

### create virtual environment
```
$ mkvirtualenv facecourse-py3 -p python3  
$ workon facecourse-py3  
```
now install python libraries within this virtual environment  

```
$ pip install numpy scipy matplotlib scikit-image scikit-learn ipython  
```

for quit virtual environment use  
```
$ deactivate  
```

## Install / Compile Opencv  
```
$ git clone https://github.com/opencv/opencv.git  
$ cd opencv  
```

### Download opencv_contrib from Github  

```
$ git clone https://github.com/opencv/opencv_contrib.git  
$ cd opencv_contrib    
```

### Cuda dependencies
comes with Nvidia Jetson Nano Ubuntu image if you need steps add TODO for this

```
$ cd ~/opencv  
$ mkdir build  
$ cd build  
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D WITH_CUDA=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_CUBLAS=1 \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..  

```

find out number of CPU cores in your machine  

```
$ nproc  
substitute 4 by output of nproc  
$ make -j4  
or then run  
$ make  
```

then  
```
$ make install  
$ sudo ldconfig  
```

### Setup python bindings  
```
$ cd /usr/local/lib/python3.6/dist-packages/cv2/python-3.6  
$ sudo mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so  

or if you are on virtual env  
$ cd ~/.virtualenvs/cv/lib/python3.6/site-packages/  
$ ln -s /usr/local/python/cv2/python-3.6/cv2.so cv2.so  
```

Test installation  

```
$ cd ~  
$ workon cv  
$ python  
Python 3.6.5 (default, Apr 1 2018, 05:46:30)  
[GCC 7.3.0] on linux  
Type "help", "copyright", "credits" or "license" for more information.  
>>> import cv2  
>>> cv2.__version__  
'3.4.4'  
>>> quit()  
```
### install precompiled Dlib  
```
$ sudo pip3 install dlib  
$ sudo pip3 install face_recognition  
```
## Compile Dlib for CUDA  
```
$ sudo apt-get update  
$ sudo apt-get install python3-dev  
$ sudo apt-get install build-dep python3  
```
### Cloning dlib  
```
git clone https://github.com/davisking/dlib.git  
```
### Building the main dlib library  
```
$ cd dlib  
$ mkdir build && cd build  

$ cmake .. -DCUDA_HOST_COMPILER=/usr/bin/gcc-6 -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/ -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -DUSE_F16C=1  
$ cmake --build . --config Release  
$ sudo ldconfig  
```

### Building and installing the Python extensions  
```
python setup.py install --record files.txt --compiler-flags "-DCUDA_HOST_COMPILER=/usr/bin/gcc-6"  
```
### Uninstalling dlib  
Once we want to uninstall dlib, simply type the command below according to generated files.txt.
```
$ cat files.txt | xargs rm -rf  
```
### Testing dlib installation  
At this point, you should be able to run python3 and type import dlib successfully.  
```
python3  
>>> import dlib  
```
### Workaround solution from NVidia for the issue on Dlib in cudnn module
Hi,  

We found a workaround to unblock this issue.  
Please use basic cudnnConvolutionForward algorithm instead.  

1. Download source  
```
wget http://dlib.net/files/dlib-19.16.tar.bz2  

tar jxvf dlib-19.16.tar.bz2  
```

2. Apply this changes:
```
diff --git a/dlib/cuda/cudnn_dlibapi.cpp b/dlib/cuda/cudnn_dlibapi.cpp  
index a32fcf6..6952584 100644  
--- a/dlib/cuda/cudnn_dlibapi.cpp  
+++ b/dlib/cuda/cudnn_dlibapi.cpp  
@@ -851,7 +851,7 @@ namespace dlib  
                         dnn_prefer_fastest_algorithms()?CUDNN_CONVOLUTION_FWD_PREFER_FASTEST:CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                         std::numeric_limits<size_t>::max(),  
                         &forward_best_algo));  
-                forward_algo = forward_best_algo;  
+                //forward_algo = forward_best_algo;  
                 CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(   
                         context(),  
                         descriptor(data),  
```

3. Build and install  
```
mkdir build  
cd build  
cmake ..  
cmake --build .  
sudo python setup.py install  
```

Our internal team keep checking the cuDNN issue and will let you know if any progress.  
Thanks.  


# How to run Application?

First make sure you download and copied trained model files then follow steps below
```
$ bash ./run.sh  
```
### if you don't want to or know how to edit .sh srcipts use command below for running on laptop. 0 is defualt device camera laptop, desktop etc. 1 is Raspberry Pi camera 2 is Jetson Nano camera module  
```
$  python3 /project root directory/ai_edge.py  --path /project root directory --cameratype 0 --target 0  
```
# Project structure  

### Scripts :
run_on_intel.sh is main application running script it first setup and load Intel VPU environment and then load application run_on_cpu.sh for running application on device CPU.

### models folder :
models folder contains both pre-trained detection models and custom trained face and object recognition models

### dataset folder :
dataset folder contains humans face dataset and will be host future datatsets

### db folder :
is a database folder just for development porpuses

### ui folder :
contans ui classes for the main application loop

### ai_edge.py :  
ai_edge.py contains init arguments live video stream object from imutils.video and main application loop

### main_application.py :  
main_application.py is main application structure class

### main_imports.py :  
contains required imports for entire application

### definitions.py :  
create for the sake of smaller main app file main_application.py import this class and use objects in it

### init_ui.py :
contains main inits to trigger ui codes

### init_buttons.py :
contains button objects for mainloop

### modules :
This folder contains all modules used in AI Edge application and will be framework structure

### face_recognition :
Folder contains three module face_pickle, face_dlib and face_scan. face_pickle uses Python pickle module to serializing and de-serializing face object (The pickle module implements binary protocols for serializing and de-serializing a Python object structure.)

### face_dlib.py :
using dlib face recognition module to recognize faces.

###face_scan.py :
scan faces and record images to human dataset folder

### ssd_yolo.py :
contains loader and runnder for Tiny Yolo object detection network

### ssd_caffe.py :
contains MobileNet network and runnder for the network in real-time

### ssd_face_cnn.py :
is copy of ssd_caffe.py with face recognition network inside copy of face_pickle.py

###open_pose.py :
 is real-time pose detection using COCO, MPI, BODY25 or HAND netowks

### open_pose_ipcam.py :
 is copy of open_pose.py is real-time pose detection using COCO, MPI, BODY25 or HAND netowks on ipcamera

### open_pose_with_ssd.py :
 is copy of open_pose.py is real-time pose detection using COCO, MPI, BODY25 or HAND netowks and it uses human detection to apply network on multple layers for more than one person

### faster_rcnn_tensorflow.py :
 contains dnn implementation of tensorflow object detection network

### edge_detection.py :
uses pre-trained caffe model EdgeNet to detect edge in real-time

### face_training_pickle.py :
extracts face embeddings from humans dataset and train network with embbedings

### squeezenet_classification.py :
OpenCV dnn implementation of DeepScale SqueezeNet in realtime
