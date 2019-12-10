echo 'Updating OS'
sudo apt-get update

echo 'Uprading OS'
sudo apt-get upgrade

echo 'Installing Dependencies'
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install unzip
sudo apt-get install pkg-config
sudo apt-get install libjpeg-dev
sudo apt-get install libpng-dev
sudo apt-get install libtiff-dev
sudo apt-get install libavcodec-dev
sudo apt-get install libavformat-dev
sudo apt-get install libswscale-dev
sudo apt-get install libv4l-dev
sudo apt-get install libxvidcore-dev
sudo apt-get install libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install gfortran
sudo apt-get install python3-dev
sudo apt-get install python3-pip
sudo apt-get install python3-tk
sudo -H pip3 install numpy
sudo -H pip3 install pillow
sudo -H pip3 install imutils
sudo -H pip3 install sklearn
sudo -H pip3 install scipy
sudo -H pip3 install matplotlib
sudo -H pip3 install scikit-image
sudo -H pip3 install scikit-learn
sudo -H pip3 install ipython
echo 'Completed installing Dependencies'

echo 'Downloading Pre-trained model files'
wget https://drive.google.com/file/d/1LUKf0doWY_3WY5c7sAw8842D3-6FMcCB/view?usp=sharing
unzip models.zip
rm models.zip
echo 'Pre-trained model files download completed'

echo '#Creating swap file if this fails use sudo dd if=/dev/zero of=/swapfile bs=1024 count=1048576'
sudo fallocate -l 4G /swapfile

echo '#Set root permission to swap file'
sudo chmod 600 /swapfile

echo '#Using using the mkswap utility to set up a Linux swap area on the file'
sudo mkswap /swapfile

echo '#Activate the swap file'
sudo swapon /swapfile
echo '/etc/fstab' >> /etc/fstab

echo 'Dlib installation cloning and building'
mkdir frameworks
cd frameworks
git clone https://github.com/davisking/dlib.git
cd dlib
cd dlib
cd cuda
echo 'we remove line containing forward_algo = forward_best_algo; in order to workaround solution dlib cudnn problem'
sed -i -e 's/forward_algo = forward_best_algo;//g' cudnn_dlibapi.cpp
cd ..
cd ..
mkdir build
cd build
cmake ..
cmake --build .
cd ..
sudo python setup.py install
cd ..
echo 'Completed Dlib Installation'

echo 'Opencv installation cloning and building'
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
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

make
make install
sudo ldconfig
find ./ -name "cv2.cpython-36m-x86_64-linux-gnu.so"
cd "$(dirname "$(!!)")"
sudo mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
find ./ -name "run_on_cpu.sh"
cd "$(dirname "$(!!)")"

rm -rf frameworks
echo 'Completed Opencv installation'
