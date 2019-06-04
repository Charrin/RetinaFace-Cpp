# push to device and run
adb push * /sdcard

# chmod
adb shell
cd /sdcard;

# su and check to root, mv files to /data
chmod 777 retina_det

# run
# ./retina_det imgpath model num_thread loop

# ncnn without opt
# 1 thread
./retina_det test.jpg mnet.ncnn 1 10
# 2 thread
./retina_det test.jpg mnet.ncnn 2 10
# 4 thread
./retina_det test.jpg mnet.ncnn 4 10

# ncnn with opt
# 1 thread
./retina_det test.jpg mnet-opt 1 10
# 2 thread
./retina_det test.jpg mnet-opt 2 10
# 4 thread
./retina_det test.jpg mnet-opt 4 10


