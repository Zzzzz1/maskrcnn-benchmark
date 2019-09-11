sudo pip3 install torch torchvision yacs tqdm pycocotools ninja cython matplotlib opencv-python
cd ..
git clone https://github.com/NVIDIA/apex.git
cd apex/
sudo rlaunch --gpu=1 --cpu=4 --memory=20480 -- python3 setup.py install --cuda_ext --cpp_ext
cd ../maskrcnn-benchmark
sudo rlaunch --gpu=1 --cpu=4 --memory=20480 -- python3 setup.py build develop
