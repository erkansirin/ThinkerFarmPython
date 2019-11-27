echo 'Installing Brew'
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

echo 'Installing Dependencies'
brew install wget
brew install libblas-dev
brew install liblapack-dev
brew install libatlas-base-dev
brew install build-essential
brew install cmake
brew install unzip
brew install pkg-config
brew install libjpeg-dev
brew install libpng-dev
brew install libtiff-dev
brew install libavcodec-dev
brew install libavformat-dev
brew install libswscale-dev
brew install libv4l-dev
brew install libxvidcore-dev
brew install libx264-dev
brew install libgtk-3-dev
brew install libatlas-base-dev
brew install gfortran
brew install python3-dev
brew install python3-pip
brew install python3-tk
pip3 install numpy
pip3 install pillow
pip3 install imutils
pip3 install sklearn
pip3 install scipy
pip3 install matplotlib
pip3 install scikit-image
pip3 install scikit-learn
pip3 install ipython

echo 'Dlib installation cloning and building'
mkdir frameworks
cd frameworks
git clone https://github.com/davisking/dlib.git
cd dlib
cd dlib
cd cuda
echo 'we remove line containing  = forward_best_algo; in order to workaround solution dlib cudnn problem'
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
