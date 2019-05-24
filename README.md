# RetinaFace-Cpp
RetinaFace detector with C++

[official RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

I convert [mobilenet-0.25 mxnet model](https://github.com/deepinsight/insightface/issues/669) (trained by [yangfly](https://github.com/yangfly)) to caffe model

* I have checked the output of the two models be the same.

* For same input images, the output of the two detector (python version and cpp version) is same.

* Haven't tested widerface yet, but it is in the plan.

**Please replace your own inference code (caffe/ncnn/feather .etc) in the source code**
